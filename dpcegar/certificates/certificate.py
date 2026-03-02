"""Certificate data structures for verification and repair proofs.

Certificates provide machine-checkable evidence that a mechanism
satisfies (or violates) a privacy guarantee.  They support
serialization, validation, and chaining for composed mechanisms.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Sequence
from pathlib import Path

import math

from dpcegar.ir.types import (
    ApproxBudget,
    FDPBudget,
    GDPBudget,
    PrivacyBudget,
    PrivacyNotion,
    PureBudget,
    RDPBudget,
    ZCDPBudget,
)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class CertificateType(Enum):
    """Kind of evidence a certificate carries."""

    VERIFICATION = auto()
    REFUTATION = auto()
    REPAIR = auto()
    COMPOSITE = auto()
    CHAIN = auto()


class ProofFormat(Enum):
    """Serialization format for the embedded proof artefact."""

    INTERNAL = auto()
    LFSC = auto()
    ALETHE = auto()
    JSON = auto()


# ---------------------------------------------------------------------------
# Base certificate
# ---------------------------------------------------------------------------


@dataclass
class Certificate:
    """Base certificate issued after verification, refutation, or repair.

    Every certificate records *which* mechanism was analysed, under
    *which* privacy notion, and a bag of ``proof_data`` that downstream
    validators can inspect.

    Parameters
    ----------
    cert_type:
        The kind of evidence this certificate carries.
    mechanism_id:
        An opaque identifier for the mechanism (e.g. a content hash).
    mechanism_name:
        Human-readable mechanism name.
    privacy_notion:
        The privacy notion the certificate speaks about.
    privacy_guarantee:
        The concrete budget that was verified / violated, if applicable.
    proof_data:
        Arbitrary proof artefact (solver traces, UNSAT cores, …).
    metadata:
        Free-form metadata (tool version, commit hash, …).
    """

    cert_type: CertificateType
    mechanism_id: str
    mechanism_name: str
    privacy_notion: PrivacyNotion
    privacy_guarantee: PrivacyBudget | None = None
    timestamp: float = field(default_factory=time.time)
    proof_data: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    valid: bool = False
    cert_id: str = field(default="", init=False)

    def __post_init__(self) -> None:
        self.cert_id = self._generate_id(
            self.mechanism_id, self.privacy_notion, self.timestamp
        )

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _generate_id(
        mechanism_id: str | None = None,
        notion: PrivacyNotion | None = None,
        ts: float | None = None,
    ) -> str:
        """Return a deterministic id from *mechanism_id*, *notion*, *ts*.

        When called without arguments (e.g. from ``__post_init__``) the
        instance attributes are **not** yet available, so the three
        values are forwarded explicitly.
        """
        raw = f"{mechanism_id}:{notion}:{ts}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def summary(self) -> str:
        """One-line human-readable summary of the certificate."""
        status = "VALID" if self.valid else "UNVALIDATED"
        guarantee = (
            str(self.privacy_guarantee) if self.privacy_guarantee else "n/a"
        )
        return (
            f"[{status}] {self.cert_type.name} certificate "
            f"for '{self.mechanism_name}' "
            f"({self.privacy_notion.name}, guarantee={guarantee})"
        )

    # -- serialization -------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a plain-dict representation suitable for JSON encoding."""
        return {
            "cert_id": self.cert_id,
            "cert_type": self.cert_type.name,
            "mechanism_id": self.mechanism_id,
            "mechanism_name": self.mechanism_name,
            "privacy_notion": self.privacy_notion.name,
            "privacy_guarantee": _budget_to_dict(self.privacy_guarantee),
            "timestamp": self.timestamp,
            "proof_data": self.proof_data,
            "metadata": self.metadata,
            "valid": self.valid,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize the certificate to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)


# ---------------------------------------------------------------------------
# Verification certificate
# ---------------------------------------------------------------------------


@dataclass
class VerificationCertificate(Certificate):
    """Certificate proving that a mechanism **satisfies** a budget.

    Includes the SMT encoding hash, an optional reference to an external
    UNSAT proof, and CEGAR-loop statistics.
    """

    encoding_hash: str = ""
    unsat_proof_ref: str | None = None
    density_bound_summary: dict[str, Any] = field(default_factory=dict)
    abstraction_depth: int = 0
    cegar_iterations: int = 0
    solver_time: float = 0.0

    def __post_init__(self) -> None:
        self.cert_type = CertificateType.VERIFICATION
        super().__post_init__()

    def validate(self) -> bool:
        """Run basic internal-consistency checks.

        Returns ``True`` when the certificate looks well-formed.  This is
        *not* a full cryptographic verification – it merely checks that
        required fields are populated and mutually consistent.
        """
        errors: list[str] = []

        if not self.encoding_hash:
            errors.append("encoding_hash is empty")

        if self.privacy_guarantee is None:
            errors.append("privacy_guarantee must be set for verification")

        if self.cegar_iterations < 0:
            errors.append("cegar_iterations must be non-negative")

        if self.solver_time < 0.0:
            errors.append("solver_time must be non-negative")

        if self.abstraction_depth < 0:
            errors.append("abstraction_depth must be non-negative")

        if not self.density_bound_summary:
            errors.append("density_bound_summary is empty")

        self.valid = len(errors) == 0
        if errors:
            self.metadata["validation_errors"] = errors
        return self.valid

    # -- factory -------------------------------------------------------------

    @classmethod
    def from_cegar_result(
        cls,
        mechanism_id: str,
        mechanism_name: str,
        result: Any,
    ) -> VerificationCertificate:
        """Build a ``VerificationCertificate`` from a :class:`CEGARResult`.

        Parameters
        ----------
        mechanism_id:
            Opaque identifier for the analysed mechanism.
        mechanism_name:
            Human-readable mechanism name.
        result:
            A ``CEGARResult`` produced by the CEGAR engine.  The type is
            kept as ``Any`` to avoid hard import cycles.
        """
        budget = getattr(result, "budget", None)
        certificate = getattr(result, "certificate", None)

        density_bounds: dict[str, Any] = {}
        abstraction_state: Any = None
        proof_steps: list[Any] = []
        if certificate is not None:
            density_bounds = getattr(certificate, "density_bounds", {}) or {}
            abstraction_state = getattr(certificate, "abstraction_state", None)
            proof_steps = getattr(certificate, "proof_steps", []) or []

        stats = getattr(result, "statistics", None)
        iterations = 0
        solver_t = 0.0
        if stats is not None:
            iterations = getattr(stats, "iterations", 0)
            solver_t = getattr(stats, "total_solver_time", 0.0)

        encoding_raw = json.dumps(
            {"mechanism": mechanism_id, "budget": str(budget)},
            sort_keys=True,
        )
        encoding_hash = hashlib.sha256(encoding_raw.encode()).hexdigest()

        notion = _infer_notion(budget)

        cert = cls(
            mechanism_id=mechanism_id,
            mechanism_name=mechanism_name,
            privacy_notion=notion,
            privacy_guarantee=budget,
            encoding_hash=encoding_hash,
            density_bound_summary=(
                density_bounds if isinstance(density_bounds, dict) else {}
            ),
            abstraction_depth=(
                len(proof_steps) if proof_steps else 0
            ),
            cegar_iterations=iterations,
            solver_time=solver_t,
            proof_data={
                "verdict": getattr(result, "verdict", None),
                "abstraction_state": str(abstraction_state),
                "proof_steps_count": len(proof_steps),
            },
        )
        cert.validate()
        return cert


# ---------------------------------------------------------------------------
# Refutation certificate
# ---------------------------------------------------------------------------


@dataclass
class RefutationCertificate(Certificate):
    """Certificate proving that a mechanism **violates** a budget.

    Carries a concrete counterexample – an assignment to the free
    variables that witnesses the privacy violation.
    """

    counterexample: dict[str, float] = field(default_factory=dict)
    violation_magnitude: float = 0.0
    path_id: int = -1

    def __post_init__(self) -> None:
        self.cert_type = CertificateType.REFUTATION
        super().__post_init__()

    def validate(self) -> bool:
        """Check that the counterexample is non-empty and magnitude > 0."""
        errors: list[str] = []

        if not self.counterexample:
            errors.append("counterexample is empty")

        if self.violation_magnitude <= 0.0:
            errors.append(
                "violation_magnitude must be positive "
                f"(got {self.violation_magnitude})"
            )

        if self.path_id < 0:
            errors.append("path_id must be non-negative")

        self.valid = len(errors) == 0
        if errors:
            self.metadata["validation_errors"] = errors
        return self.valid

    def to_dict(self) -> dict[str, Any]:
        """Extend base dict with refutation-specific fields."""
        d = super().to_dict()
        d.update(
            {
                "counterexample": self.counterexample,
                "violation_magnitude": self.violation_magnitude,
                "path_id": self.path_id,
            }
        )
        return d


# ---------------------------------------------------------------------------
# Repair certificate
# ---------------------------------------------------------------------------


@dataclass
class RepairCertificate(Certificate):
    """Certificate attesting to a successful (or attempted) repair.

    Records what changed, the associated cost, and – when available – a
    :class:`VerificationCertificate` proving that the repaired mechanism
    satisfies the target budget.
    """

    original_mechanism_id: str = ""
    repair_description: str = ""
    parameter_changes: dict[str, tuple[float, float]] = field(
        default_factory=dict,
    )
    repair_cost: float = 0.0
    verification_cert: VerificationCertificate | None = None

    def __post_init__(self) -> None:
        self.cert_type = CertificateType.REPAIR
        super().__post_init__()

    def validate(self) -> bool:
        """Validate the repair certificate.

        Checks that the repair description is non-empty, that at least
        one parameter was changed, and that the optional verification
        sub-certificate is itself valid.
        """
        errors: list[str] = []

        if not self.original_mechanism_id:
            errors.append("original_mechanism_id is empty")

        if not self.repair_description:
            errors.append("repair_description is empty")

        if not self.parameter_changes:
            errors.append("parameter_changes is empty")

        if self.repair_cost < 0.0:
            errors.append("repair_cost must be non-negative")

        if self.verification_cert is not None:
            if not self.verification_cert.valid:
                if not self.verification_cert.validate():
                    errors.append(
                        "embedded verification certificate is invalid"
                    )

        self.valid = len(errors) == 0
        if errors:
            self.metadata["validation_errors"] = errors
        return self.valid

    def to_dict(self) -> dict[str, Any]:
        """Extend base dict with repair-specific fields."""
        d = super().to_dict()
        d.update(
            {
                "original_mechanism_id": self.original_mechanism_id,
                "repair_description": self.repair_description,
                "parameter_changes": {
                    k: list(v) for k, v in self.parameter_changes.items()
                },
                "repair_cost": self.repair_cost,
                "verification_cert": (
                    self.verification_cert.to_dict()
                    if self.verification_cert is not None
                    else None
                ),
            }
        )
        return d

    # -- factory -------------------------------------------------------------

    @classmethod
    def from_repair_result(
        cls,
        mechanism_id: str,
        mechanism_name: str,
        result: Any,
    ) -> RepairCertificate:
        """Build a ``RepairCertificate`` from a ``RepairResult``.

        Parameters
        ----------
        mechanism_id:
            Identifier of the **original** (pre-repair) mechanism.
        mechanism_name:
            Human-readable mechanism name.
        result:
            A ``RepairResult`` from the CEGIS repair synthesizer.
        """
        param_values: dict[str, float] = (
            getattr(result, "parameter_values", {}) or {}
        )
        template = getattr(result, "template", None)
        template_name = getattr(template, "name", "unknown") if template else "unknown"

        original_params: dict[str, float] = {}
        if template is not None:
            original_params = getattr(template, "original_values", {}) or {}

        changes: dict[str, tuple[float, float]] = {}
        for pname, new_val in param_values.items():
            old_val = original_params.get(pname, 0.0)
            if old_val != new_val:
                changes[pname] = (old_val, new_val)

        cost = getattr(result, "repair_cost", 0.0)

        ver_cert_raw = getattr(result, "verification_certificate", None)
        ver_cert: VerificationCertificate | None = None
        if ver_cert_raw is not None and isinstance(
            ver_cert_raw, VerificationCertificate
        ):
            ver_cert = ver_cert_raw

        budget = getattr(result, "budget", None) or (
            getattr(ver_cert, "privacy_guarantee", None)
            if ver_cert
            else None
        )

        notion = _infer_notion(budget)

        cert = cls(
            mechanism_id=mechanism_id,
            mechanism_name=mechanism_name,
            privacy_notion=notion,
            privacy_guarantee=budget,
            original_mechanism_id=mechanism_id,
            repair_description=f"CEGIS repair via template '{template_name}'",
            parameter_changes=changes,
            repair_cost=cost,
            verification_cert=ver_cert,
            proof_data={
                "verdict": str(getattr(result, "verdict", None)),
                "template": template_name,
            },
        )
        cert.validate()
        return cert


# ---------------------------------------------------------------------------
# Composite certificate (multi-notion)
# ---------------------------------------------------------------------------


@dataclass
class CompositeCertificate(Certificate):
    """Certificate aggregating results across multiple privacy notions.

    A mechanism may simultaneously satisfy pure-DP, approx-DP, and zCDP
    guarantees.  A ``CompositeCertificate`` groups the per-notion
    certificates and any derived cross-notion guarantees.
    """

    variant_certificates: dict[PrivacyNotion, Certificate] = field(
        default_factory=dict,
    )
    derived_guarantees: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.cert_type = CertificateType.COMPOSITE
        super().__post_init__()

    # -- mutation ------------------------------------------------------------

    def add_variant(
        self, notion: PrivacyNotion, cert: Certificate
    ) -> None:
        """Register a per-notion certificate."""
        self.variant_certificates[notion] = cert

    def get_variant(self, notion: PrivacyNotion) -> Certificate | None:
        """Retrieve the certificate for *notion*, or ``None``."""
        return self.variant_certificates.get(notion)

    # -- queries -------------------------------------------------------------

    def all_verified(self) -> bool:
        """Return ``True`` iff every variant certificate is valid (vacuously true if empty)."""
        return all(c.valid for c in self.variant_certificates.values())

    def verified_notions(self) -> list[PrivacyNotion]:
        """Return the notions whose certificates are valid."""
        return [
            n
            for n, c in self.variant_certificates.items()
            if c.valid
        ]

    # -- serialization -------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d["variant_certificates"] = {
            n.name: c.to_dict() for n, c in self.variant_certificates.items()
        }
        d["derived_guarantees"] = self.derived_guarantees
        return d


# ---------------------------------------------------------------------------
# Certificate chain (sequential / parallel composition)
# ---------------------------------------------------------------------------


@dataclass
class CertificateChain:
    """An ordered list of certificates that together cover a composed mechanism.

    The chain records the composition strategy (``"sequential"`` or
    ``"parallel"``) and can compute the overall composed privacy
    guarantee.
    """

    chain_id: str = ""
    certificates: list[Certificate] = field(default_factory=list)
    composition_type: str = "sequential"
    composed_guarantee: PrivacyBudget | None = None

    def __post_init__(self) -> None:
        if not self.chain_id:
            raw = f"chain:{id(self)}:{time.time()}"
            self.chain_id = hashlib.sha256(raw.encode()).hexdigest()[:16]

    # -- mutation ------------------------------------------------------------

    def add(self, cert: Certificate) -> None:
        """Append a certificate to the chain."""
        self.certificates.append(cert)

    # -- validation ----------------------------------------------------------

    def validate_chain(self) -> bool:
        """Check that every certificate is valid and budgets compose.

        For *sequential* composition of pure-DP budgets the total
        epsilon must equal the sum of per-step epsilons.  For
        *parallel* composition the total is the maximum.
        """
        if not self.certificates:
            return False

        for cert in self.certificates:
            if not cert.valid:
                return False

        if self.composed_guarantee is None:
            return True

        computed = self.composed_budget()
        if computed is None:
            return True

        if isinstance(computed, PureBudget) and isinstance(
            self.composed_guarantee, PureBudget
        ):
            return abs(computed.epsilon - self.composed_guarantee.epsilon) < 1e-12

        if isinstance(computed, ApproxBudget) and isinstance(
            self.composed_guarantee, ApproxBudget
        ):
            return (
                abs(computed.epsilon - self.composed_guarantee.epsilon) < 1e-12
                and abs(computed.delta - self.composed_guarantee.delta) < 1e-15
            )

        if isinstance(computed, ZCDPBudget) and isinstance(
            self.composed_guarantee, ZCDPBudget
        ):
            return abs(computed.rho - self.composed_guarantee.rho) < 1e-12

        if isinstance(computed, RDPBudget) and isinstance(
            self.composed_guarantee, RDPBudget
        ):
            return (
                computed.alpha == self.composed_guarantee.alpha
                and abs(computed.epsilon - self.composed_guarantee.epsilon) < 1e-12
            )

        if isinstance(computed, GDPBudget) and isinstance(
            self.composed_guarantee, GDPBudget
        ):
            return abs(computed.mu - self.composed_guarantee.mu) < 1e-12

        if isinstance(computed, FDPBudget) and isinstance(
            self.composed_guarantee, FDPBudget
        ):
            for alpha in [i / 1000 for i in range(1001)]:
                if computed.trade_off_fn(alpha) < self.composed_guarantee.trade_off_fn(alpha) - 1e-9:
                    return False
            return True

        return True

    # -- composition ---------------------------------------------------------

    def composed_budget(self) -> PrivacyBudget | None:
        """Derive the composed budget from the chain certificates.

        Only handles :class:`PureBudget` and :class:`ApproxBudget` for
        sequential and parallel composition.  Returns ``None`` for
        unsupported combinations.
        """
        budgets = [
            c.privacy_guarantee
            for c in self.certificates
            if c.privacy_guarantee is not None
        ]
        if not budgets:
            return None

        if all(isinstance(b, PureBudget) for b in budgets):
            epsilons = [b.epsilon for b in budgets]  # type: ignore[union-attr]
            if self.composition_type == "sequential":
                return PureBudget(epsilon=sum(epsilons))
            return PureBudget(epsilon=max(epsilons))

        if all(isinstance(b, ApproxBudget) for b in budgets):
            epsilons = [b.epsilon for b in budgets]  # type: ignore[union-attr]
            deltas = [b.delta for b in budgets]  # type: ignore[union-attr]
            if self.composition_type == "sequential":
                return ApproxBudget(
                    epsilon=sum(epsilons), delta=sum(deltas)
                )
            return ApproxBudget(
                epsilon=max(epsilons), delta=max(deltas)
            )

        if all(isinstance(b, ZCDPBudget) for b in budgets):
            rhos = [b.rho for b in budgets]  # type: ignore[union-attr]
            if self.composition_type == "sequential":
                return ZCDPBudget(rho=sum(rhos))
            return ZCDPBudget(rho=max(rhos))

        if all(isinstance(b, RDPBudget) for b in budgets):
            alphas = [b.alpha for b in budgets]  # type: ignore[union-attr]
            if len(set(alphas)) != 1:
                return None
            epsilons = [b.epsilon for b in budgets]  # type: ignore[union-attr]
            alpha = alphas[0]
            if self.composition_type == "sequential":
                return RDPBudget(alpha=alpha, epsilon=sum(epsilons))
            return RDPBudget(alpha=alpha, epsilon=max(epsilons))

        if all(isinstance(b, GDPBudget) for b in budgets):
            mus = [b.mu for b in budgets]  # type: ignore[union-attr]
            if self.composition_type == "sequential":
                return GDPBudget(mu=math.sqrt(sum(m ** 2 for m in mus)))
            return GDPBudget(mu=max(mus))

        if all(isinstance(b, FDPBudget) for b in budgets):
            if self.composition_type == "sequential":
                from functools import reduce
                return reduce(lambda a, b: a.compose(b), budgets)  # type: ignore[arg-type]
            else:
                fns = [b.trade_off_fn for b in budgets]  # type: ignore[union-attr]

                def parallel_tradeoff(alpha: float) -> float:
                    return max(fn(alpha) for fn in fns)

                return FDPBudget(trade_off_fn=parallel_tradeoff)

        return None

    # -- serialization -------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary."""
        return {
            "chain_id": self.chain_id,
            "composition_type": self.composition_type,
            "composed_guarantee": _budget_to_dict(self.composed_guarantee),
            "certificates": [c.to_dict() for c in self.certificates],
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize the chain to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


@dataclass
class ValidationResult:
    """Outcome of a certificate or chain validation."""

    valid: bool = False
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.valid

    def summary(self) -> str:
        """Human-readable one-liner."""
        tag = "PASS" if self.valid else "FAIL"
        parts = [f"[{tag}]"]
        if self.errors:
            parts.append(f"{len(self.errors)} error(s)")
        if self.warnings:
            parts.append(f"{len(self.warnings)} warning(s)")
        return " ".join(parts)


class CertificateValidator:
    """Stateless validator for certificates and chains.

    Each ``validate`` call returns a :class:`ValidationResult` that
    collects errors and warnings independently.
    """

    # -- public API ----------------------------------------------------------

    def validate(self, cert: Certificate) -> ValidationResult:
        """Run all validation checks on a single certificate.

        Checks budget consistency, proof-reference presence, and
        timestamp plausibility.
        """
        errors: list[str] = []
        warnings: list[str] = []

        if not cert.cert_id:
            errors.append("cert_id is empty")

        if not cert.mechanism_id:
            errors.append("mechanism_id is empty")

        if not self._check_timestamp(cert):
            warnings.append(
                "timestamp is in the future or unreasonably old"
            )

        if not self._check_budget_consistency(cert):
            errors.append(
                "budget type does not match the declared privacy notion"
            )

        if not self._check_proof_reference(cert):
            warnings.append("proof_data is empty – certificate has no proof")

        valid = len(errors) == 0
        return ValidationResult(valid=valid, errors=errors, warnings=warnings)

    def validate_chain(self, chain: CertificateChain) -> ValidationResult:
        """Validate every certificate in the chain and check composition."""
        errors: list[str] = []
        warnings: list[str] = []

        if not chain.certificates:
            errors.append("chain contains no certificates")
            return ValidationResult(
                valid=False, errors=errors, warnings=warnings
            )

        for idx, cert in enumerate(chain.certificates):
            sub = self.validate(cert)
            for e in sub.errors:
                errors.append(f"certificate[{idx}]: {e}")
            for w in sub.warnings:
                warnings.append(f"certificate[{idx}]: {w}")

        if chain.composition_type not in ("sequential", "parallel"):
            errors.append(
                f"unknown composition_type '{chain.composition_type}'"
            )

        if not chain.validate_chain():
            errors.append("chain.validate_chain() returned False")

        valid = len(errors) == 0
        return ValidationResult(valid=valid, errors=errors, warnings=warnings)

    # -- internal checks -----------------------------------------------------

    @staticmethod
    def _check_budget_consistency(cert: Certificate) -> bool:
        """Return ``True`` when the budget type matches the privacy notion."""
        if cert.privacy_guarantee is None:
            return True

        mapping: dict[PrivacyNotion, type] = {
            PrivacyNotion.PURE_DP: PureBudget,
            PrivacyNotion.APPROX_DP: ApproxBudget,
        }
        expected = mapping.get(cert.privacy_notion)
        if expected is None:
            return True
        return isinstance(cert.privacy_guarantee, expected)

    @staticmethod
    def _check_proof_reference(cert: Certificate) -> bool:
        """Return ``True`` when the certificate carries some proof data."""
        return bool(cert.proof_data)

    @staticmethod
    def _check_timestamp(cert: Certificate) -> bool:
        """Return ``True`` when the timestamp is plausible.

        A timestamp is considered plausible when it is not more than
        60 seconds in the future and not older than ten years.
        """
        now = time.time()
        ten_years = 10 * 365.25 * 24 * 3600
        return (cert.timestamp <= now + 60) and (
            cert.timestamp >= now - ten_years
        )


# ---------------------------------------------------------------------------
# Serializer / deserializer
# ---------------------------------------------------------------------------


class CertificateSerializer:
    """JSON-based (de)serializer for certificate objects.

    Round-tripping is supported for the concrete certificate subtypes
    :class:`VerificationCertificate`, :class:`RefutationCertificate`,
    :class:`RepairCertificate`, and :class:`CompositeCertificate`.
    """

    _TYPE_MAP: dict[str, type[Certificate]] = {
        CertificateType.VERIFICATION.name: VerificationCertificate,
        CertificateType.REFUTATION.name: RefutationCertificate,
        CertificateType.REPAIR.name: RepairCertificate,
        CertificateType.COMPOSITE.name: CompositeCertificate,
    }

    # -- public API ----------------------------------------------------------

    def serialize(self, cert: Certificate) -> str:
        """Serialize *cert* to a JSON string."""
        return cert.to_json()

    def deserialize(self, data: str) -> Certificate:
        """Deserialize a JSON string into the appropriate certificate type.

        Raises
        ------
        ValueError
            When the ``cert_type`` field is missing or unrecognised.
        """
        raw: dict[str, Any] = json.loads(data)
        cert_type_name = raw.get("cert_type")
        if cert_type_name is None:
            raise ValueError("JSON has no 'cert_type' field")

        cls = self._TYPE_MAP.get(cert_type_name)
        if cls is None:
            raise ValueError(f"unknown cert_type '{cert_type_name}'")

        notion = PrivacyNotion[raw["privacy_notion"]]
        budget = self._dict_to_budget(raw.get("privacy_guarantee"))

        base_kwargs: dict[str, Any] = {
            "cert_type": CertificateType[cert_type_name],
            "mechanism_id": raw["mechanism_id"],
            "mechanism_name": raw["mechanism_name"],
            "privacy_notion": notion,
            "privacy_guarantee": budget,
            "timestamp": raw.get("timestamp", time.time()),
            "proof_data": raw.get("proof_data", {}),
            "metadata": raw.get("metadata", {}),
            "valid": raw.get("valid", False),
        }

        if cls is VerificationCertificate:
            return cls(
                **base_kwargs,
                encoding_hash=raw.get("encoding_hash", ""),
                unsat_proof_ref=raw.get("unsat_proof_ref"),
                density_bound_summary=raw.get("density_bound_summary", {}),
                abstraction_depth=raw.get("abstraction_depth", 0),
                cegar_iterations=raw.get("cegar_iterations", 0),
                solver_time=raw.get("solver_time", 0.0),
            )

        if cls is RefutationCertificate:
            return cls(
                **base_kwargs,
                counterexample=raw.get("counterexample", {}),
                violation_magnitude=raw.get("violation_magnitude", 0.0),
                path_id=raw.get("path_id", -1),
            )

        if cls is RepairCertificate:
            changes_raw = raw.get("parameter_changes", {})
            changes = {
                k: tuple(v) for k, v in changes_raw.items()
            }
            ver_cert_raw = raw.get("verification_cert")
            ver_cert: VerificationCertificate | None = None
            if ver_cert_raw is not None:
                ver_cert = self.deserialize(  # type: ignore[assignment]
                    json.dumps(ver_cert_raw)
                )
            return cls(
                **base_kwargs,
                original_mechanism_id=raw.get("original_mechanism_id", ""),
                repair_description=raw.get("repair_description", ""),
                parameter_changes=changes,  # type: ignore[arg-type]
                repair_cost=raw.get("repair_cost", 0.0),
                verification_cert=ver_cert,
            )

        if cls is CompositeCertificate:
            variant_certs: dict[PrivacyNotion, Certificate] = {}
            for notion_name, sub_dict in raw.get(
                "variant_certificates", {}
            ).items():
                sub_cert = self.deserialize(json.dumps(sub_dict))
                variant_certs[PrivacyNotion[notion_name]] = sub_cert
            return cls(
                **base_kwargs,
                variant_certificates=variant_certs,
                derived_guarantees=raw.get("derived_guarantees", []),
            )

        return cls(**base_kwargs)  # pragma: no cover

    # -- file I/O ------------------------------------------------------------

    def save(self, cert: Certificate, path: Path) -> None:
        """Write the certificate to *path* as JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.serialize(cert), encoding="utf-8")

    def load(self, path: Path) -> Certificate:
        """Load a certificate from a JSON file at *path*."""
        data = path.read_text(encoding="utf-8")
        return self.deserialize(data)

    # -- budget helpers ------------------------------------------------------

    @staticmethod
    def _budget_to_dict(budget: PrivacyBudget | None) -> dict[str, Any] | None:
        """Convert a budget to a JSON-friendly dict."""
        return _budget_to_dict(budget)

    @staticmethod
    def _dict_to_budget(
        data: dict[str, Any] | None,
    ) -> PrivacyBudget | None:
        """Reconstruct a :class:`PrivacyBudget` from a dict."""
        return _dict_to_budget(data)


# ---------------------------------------------------------------------------
# Module-level helpers (shared by Certificate and CertificateSerializer)
# ---------------------------------------------------------------------------


def _budget_to_dict(budget: PrivacyBudget | None) -> dict[str, Any] | None:
    """Convert a :class:`PrivacyBudget` to a plain dict, or ``None``."""
    if budget is None:
        return None

    if isinstance(budget, PureBudget):
        return {"type": "PureBudget", "epsilon": budget.epsilon}

    if isinstance(budget, ApproxBudget):
        return {
            "type": "ApproxBudget",
            "epsilon": budget.epsilon,
            "delta": budget.delta,
        }

    # Fallback: store whatever attributes the budget exposes.
    attrs = {
        k: v
        for k, v in vars(budget).items()
        if not k.startswith("_")
    }
    return {"type": type(budget).__name__, **attrs}


def _dict_to_budget(data: dict[str, Any] | None) -> PrivacyBudget | None:
    """Reconstruct a :class:`PrivacyBudget` from a dict produced by
    :func:`_budget_to_dict`.
    """
    if data is None:
        return None

    budget_type = data.get("type", "")

    if budget_type == "PureBudget":
        return PureBudget(epsilon=data["epsilon"])

    if budget_type == "ApproxBudget":
        return ApproxBudget(epsilon=data["epsilon"], delta=data["delta"])

    return None


def _infer_notion(budget: PrivacyBudget | None) -> PrivacyNotion:
    """Best-effort mapping from a budget object to a :class:`PrivacyNotion`."""
    if budget is None:
        return PrivacyNotion.PURE_DP

    if isinstance(budget, PureBudget):
        return PrivacyNotion.PURE_DP

    if isinstance(budget, ApproxBudget):
        return PrivacyNotion.APPROX_DP

    type_name = type(budget).__name__.upper()
    for notion in PrivacyNotion:
        if notion.name in type_name:
            return notion

    return PrivacyNotion.PURE_DP
