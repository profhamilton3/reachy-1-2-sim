"""Broader phrasing, the same typed actions (#92).

`panel_planner` is a deterministic parser and stays one.  "Rest", "stow",
"wave" and "point" do not need a trained model to recognise the phrases an
operator actually types, and the registry that recognises them is inspectable,
testable and offline.  This module is an OPTIONAL layer that lets the operator
phrase a request some other way — "could you put your arm away now?" — and
turns it into the same thing the registry would have produced: an ability name
and its arguments.

OFF BY DEFAULT.  With `REACHY_PANEL_LANGUAGE` unset the panel behaves exactly
as it did before this module existed, and no request leaves the machine.

WHERE IT SITS, AND WHY THAT IS THE WHOLE SAFETY ARGUMENT
---------------------------------------------------------
It runs LAST, after every deterministic path has already declined.  That
ordering is not a performance choice:

  * `abilities.match` raises `AbilityRefusal` for a negated request ("don't
    wave"), a compound one ("wave and then stow") and for "reset the
    simulation".  Those are answers, not failures to understand, and the
    planner returns them before this module is reached.  A model asked to
    interpret "don't wave" would very reasonably answer `wave`.
  * The pick-and-place grammar still wins wherever it matches, so the
    behaviour of every command that works today is untouched.
  * `_JOINT_RE` has already refused the operator's joint angles, and the same
    expression — `abilities.JOINT_RE`, one copy — refuses the model's.

WHAT COMES BACK IS A NAME AND SOME STRINGS, AND IS TREATED AS HOSTILE
---------------------------------------------------------------------
The action name must be in `REGISTRY`; the argument names must be slots that
ability declares; the argument values must not look like joint angles.  Values
are then resolved by the existing scene layer exactly as a typed command's
are — the model does not get to assert that an object is on the board, or
where.  Anything that fails is `unsupported`, not an attempt.

THE MODEL'S PROSE IS NEVER SHOWN TO THE OPERATOR.  The card says what the
REGISTRY says the ability does.  A model that returned a summary would be
writing the sentence the operator confirms against, and a request to move an
arm is not the place to find out whether it wrote it honestly.

NOTHING ABOUT WHAT THE ARM DOES CHANGES.  Confirmation is still required for
everything that moves, the deterministic executor still flies the route, and
the proposal records which path produced it.

NO CREDENTIAL IS LOGGED, STORED OR RETURNED.  The key is read from the
environment at call time and never leaves this module: not into a log line, not
into a `Proposal`, not into an episode record, and not into anything the
browser can read.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Tuple

import panel_abilities as abilities

log = logging.getLogger("panel.language")

#: How the panel names the two interpretations, in the proposal and on the card.
BY_REGISTRY = "registry"
BY_MODEL = "language-model"

#: Below this the answer is "I did not understand that".  A model that is
#: unsure about which of two arm movements was meant should say so by being
#: unsure, and the deterministic answer to an unrecognised command is already
#: a good one.
DEFAULT_MIN_CONFIDENCE = 0.75

#: Seconds.  A human is waiting on this, and the fallback is immediate.
DEFAULT_TIMEOUT_S = 6.0

#: The default model.  `REACHY_PANEL_LANGUAGE_MODEL` overrides it; Haiku 4.5
#: (`claude-haiku-4-5-20251001`) is the choice to make if the round trip is too
#: slow on the planning path, at some cost in phrasing coverage.
DEFAULT_MODEL = "claude-sonnet-5"

_API_URL = "https://api.anthropic.com/v1/messages"
_API_VERSION = "2023-06-01"

#: Kept small on purpose: the output is a name and a couple of short strings,
#: and a larger budget only buys room for prose nobody reads.
_MAX_TOKENS = 256


@dataclasses.dataclass(frozen=True)
class Action:
    """One typed action, already checked against the registry.

    There is no field for the model's own description of what it did.  That is
    deliberate — see the module docstring.
    """

    ability: str
    arguments: Dict[str, str] = dataclasses.field(default_factory=dict)
    confidence: float = 0.0

    def as_command(self) -> str:
        """The action as a command the deterministic parser already handles.

        THE MODEL'S OUTPUT RE-ENTERS THROUGH THE FRONT DOOR.  Rather than
        constructing a match object and skipping the registry, the action is
        written back out as a canonical phrase and handed to `abilities.match`,
        which applies every check it applies to anything an operator types —
        full-intent matching, negation, compounds, the arm rule.  A path that
        bypassed those would be a second, less careful parser.
        """
        arm = self.arguments.get(abilities.SLOT_ARM, "")
        cell = self.arguments.get(abilities.SLOT_CELL, "")
        obj = self.arguments.get(abilities.SLOT_OBJECT, "")
        if self.ability == "greet":
            return "hello"
        if self.ability == "rest_forearm":
            return f"rest {arm + ' ' if arm else ''}forearm".strip()
        if self.ability == "stow_arm":
            return f"stow {arm + ' ' if arm else ''}arm".strip()
        if self.ability == "wave":
            return f"wave {arm + ' ' if arm else ''}hand".strip() if arm else "wave"
        if self.ability == "point_cell":
            return f"point to {cell}".strip()
        if self.ability == "point_object":
            return f"point to {obj}".strip()
        return ""


class Provider:
    """Something that turns a sentence into a raw JSON string."""

    def complete(self, prompt: str, text: str, timeout_s: float) -> str:
        raise NotImplementedError


class AnthropicProvider(Provider):
    """The Messages API over `urllib`, so the image gains no dependency.

    The key is read per call from the environment and held in a local.  It is
    never logged, never returned, and never stored on this object — an
    attribute would survive into a traceback or a `repr` of the adapter.
    """

    def __init__(self, model: str = "", api_url: str = "") -> None:
        self.model = model or os.environ.get(
            "REACHY_PANEL_LANGUAGE_MODEL", "") or DEFAULT_MODEL
        self._api_url = api_url or _API_URL

    def complete(self, prompt: str, text: str, timeout_s: float) -> str:
        key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            raise RuntimeError("no API key in the environment")

        body = json.dumps({
            "model": self.model,
            "max_tokens": _MAX_TOKENS,
            "system": prompt,
            "messages": [{"role": "user", "content": text}],
        }).encode("utf-8")

        request = urllib.request.Request(
            self._api_url, data=body, method="POST",
            headers={"content-type": "application/json",
                     "anthropic-version": _API_VERSION,
                     "x-api-key": key})
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            payload = json.loads(response.read().decode("utf-8"))

        parts = payload.get("content") or []
        return "".join(p.get("text", "") for p in parts
                       if isinstance(p, dict) and p.get("type") == "text")


def build_prompt() -> str:
    """The instructions, generated from the registry rather than transcribed.

    A hand-written list of abilities here would be a second copy of the
    registry, free to disagree with it — and the disagreement would show up as
    a model confidently naming an action the panel then refuses.
    """
    lines = [
        "You map a robot operator's sentence onto exactly one action from a "
        "fixed list, or onto none.",
        "",
        "Reply with a single JSON object and nothing else:",
        '  {"action": "<name or empty>", "arguments": {}, "confidence": 0.0}',
        "",
        "The actions, and the arguments each one accepts:",
    ]
    for name, ability in abilities.REGISTRY.items():
        slots = ", ".join(ability.slots) if ability.slots else "none"
        lines.append(f'  - "{name}": {ability.summary}. arguments: {slots}')
    lines += [
        "",
        "Rules:",
        "  - Use \"\" for the action when the sentence is not one of these, "
        "when it asks for two things, or when it asks for something NOT to be "
        "done. Do not guess.",
        "  - which_cell must look like r2c2. which_object is the operator's "
        "own word for the object; do not invent an object or assert where "
        "anything is.",
        "  - which_arm is 'left' or 'right', and only if the operator said so.",
        "  - Never output joint names, joint angles, degrees, radians or a "
        "trajectory. There is no action that accepts them.",
        "  - confidence is your own, between 0 and 1. Be unsure when you are.",
    ]
    return "\n".join(lines)


class LanguageAdapter:
    """Interprets phrasing the registry did not recognise.  Never raises."""

    def __init__(self, provider: Provider, *,
                 min_confidence: float = DEFAULT_MIN_CONFIDENCE,
                 timeout_s: float = DEFAULT_TIMEOUT_S) -> None:
        self._provider = provider
        self._min_confidence = min_confidence
        self._timeout_s = timeout_s
        self._prompt = build_prompt()

    def interpret(self, text: str) -> Tuple[Optional[Action], str]:
        """The action this sentence asks for, and why there is none if there is
        not.

        The second value is for the log and for a test.  It is NOT shown to the
        operator: "the model returned an unknown action" is a fact about the
        panel's internals, and the operator asked about their arm.
        """
        try:
            raw = self._provider.complete(self._prompt, text, self._timeout_s)
        except (urllib.error.URLError, OSError, RuntimeError, ValueError) as exc:
            # The class of failure, never the message: a URL error's string
            # carries the host, and an auth failure's can carry the header.
            log.info("language adapter unavailable (%s); using the registry",
                     type(exc).__name__)
            return None, f"the adapter was unavailable ({type(exc).__name__})"
        except Exception:                      # noqa: BLE001
            log.exception("language adapter failed; using the registry")
            return None, "the adapter failed"

        return validate(raw, min_confidence=self._min_confidence)


def validate(raw: str, *, min_confidence: float = DEFAULT_MIN_CONFIDENCE
             ) -> Tuple[Optional[Action], str]:
    """Turn a model's reply into an Action, or explain why it is not one.

    Every check here assumes the reply is hostile, because the cheapest way to
    find out it was not is to treat it as though it were.  A reply is a name
    and some short strings; anything else about it is discarded, including any
    instructions it may contain.
    """
    text = (raw or "").strip()
    if not text:
        return None, "the model returned nothing"

    # A model that wraps JSON in prose or a fence has still answered; a model
    # that returned prose with a brace in it has not, and the strict parse
    # below is what tells them apart.
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None, "the model did not return an object"
    try:
        payload = json.loads(text[start:end + 1])
    except ValueError:
        return None, "the model's object did not parse"
    if not isinstance(payload, dict):
        return None, "the model did not return an object"

    name = payload.get("action")
    if not isinstance(name, str) or not name:
        return None, "the model named no action"
    ability = abilities.REGISTRY.get(name)
    if ability is None:
        # Refused by NAME before anything else is read.  This is the check the
        # issue asks for a hostile test against, and it is deliberately an
        # exact lookup in the registry rather than a pattern.
        return None, f"'{name}' is not an action this robot has"

    raw_args = payload.get("arguments")
    if raw_args is None:
        raw_args = {}
    if not isinstance(raw_args, dict):
        return None, "the model's arguments were not an object"

    arguments: Dict[str, str] = {}
    for key, value in raw_args.items():
        if key not in ability.slots:
            return None, (f"'{name}' does not take an argument called "
                          f"'{key}'")
        if not isinstance(value, (str, int, float)) or isinstance(value, bool):
            return None, f"the value for '{key}' is not a word"
        as_text = str(value).strip()
        if not as_text:
            continue
        if abilities.JOINT_RE.search(as_text):
            # The same refusal the operator's own text gets, on the same
            # expression.  A model emitting anything resembling a trajectory is
            # a bug, not a capability.
            return None, f"the value for '{key}' names a joint or an angle"
        if len(as_text) > 64:
            return None, f"the value for '{key}' is too long to be a name"
        arguments[key] = as_text

    try:
        confidence = float(payload.get("confidence", 0.0))
    except (TypeError, ValueError):
        return None, "the model's confidence did not parse"
    if not 0.0 <= confidence <= 1.0:
        return None, "the model's confidence was not between 0 and 1"
    if confidence < min_confidence:
        return None, (f"the model was {confidence:.2f} sure, and "
                      f"{min_confidence:.2f} is the bar")

    action = Action(ability=name, arguments=arguments, confidence=confidence)
    if not action.as_command():
        return None, f"'{name}' has no phrasing this panel can re-read"
    return action, ""


def build_adapter() -> Optional[LanguageAdapter]:
    """The adapter for this deployment, or None — which is the default.

    OFF UNLESS SWITCHED ON, and off again if there is no key.  The alternative
    — on by default, failing on every command until somebody sets a variable —
    would put a per-command round trip and a per-command failure between the
    operator and a panel that worked fine without it.
    """
    if os.environ.get("REACHY_PANEL_LANGUAGE", "0").lower() not in (
            "1", "true", "yes", "on"):
        return None
    if not os.environ.get("ANTHROPIC_API_KEY", ""):
        log.warning("REACHY_PANEL_LANGUAGE is on but no API key is set; "
                    "the panel will use the registry only")
        return None
    return LanguageAdapter(
        AnthropicProvider(),
        min_confidence=_float_env("REACHY_PANEL_LANGUAGE_MIN_CONFIDENCE",
                                  DEFAULT_MIN_CONFIDENCE),
        timeout_s=_float_env("REACHY_PANEL_LANGUAGE_TIMEOUT_S",
                             DEFAULT_TIMEOUT_S))


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except ValueError:
        return default
