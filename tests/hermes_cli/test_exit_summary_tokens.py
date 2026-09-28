"""The one-shot exit output's token line is a scraped contract, not cosmetics.

A wrapper that runs Hermes as a subprocess (the Paperclip ``hermes_local`` adapter is the
concrete consumer) has no access to ``state.db``; it attributes a run's spend by regexing the
process output. Before this line existed, every such run recorded zero tokens while the API
calls were real. The regex below is pinned verbatim from the consumer so that if it ever
changes shape, this file fails loudly instead of spend attribution going silently to zero
again in production.

Both one-shot paths are covered on purpose: ``-q`` prints the summary via
``_print_exit_summary``, while ``-Q`` never calls it — so a wrapper flipping quiet mode on
must not silently undo the fix.
"""

from __future__ import annotations

import re
from datetime import datetime
from types import SimpleNamespace

import cli
from hermes_cli.cli_session_mixin import format_token_usage_line

# Pinned from the consumer: @paperclipai/hermes-paperclip-adapter
# dist/shared/constants.js -> TOKEN_USAGE_REGEX. Applied against stdout+stderr combined.
CONSUMER_TOKEN_USAGE_REGEX = re.compile(
    r"tokens?[:\s]+(\d+)\s*(?:input|in)\b.*?(\d+)\s*(?:output|out)\b", re.IGNORECASE)


def _agent(input_tokens, output_tokens):
    return SimpleNamespace(
        session_input_tokens=input_tokens, session_output_tokens=output_tokens)


def _scraped(text):
    """(input, output) as the consumer would read them out of a run's output, or None."""
    match = CONSUMER_TOKEN_USAGE_REGEX.search(text)
    return (int(match.group(1)), int(match.group(2))) if match else None


class TestTokenUsageLineIsScrapable:
    def test_the_consumer_reads_back_exactly_the_agents_counters(self):
        line = format_token_usage_line(_agent(40329, 2899))
        assert _scraped(line) == (40329, 2899)

    def test_six_figure_counts_survive_unformatted(self):
        """Thousands separators would truncate the capture: ``(\\d+)`` stops at the comma, so
        "105,432 input" is scraped as 432. The line must stay plain digits."""
        line = format_token_usage_line(_agent(105432, 14877))
        # The field separator comma is fine; a comma *between digits* is the hazard.
        assert not re.search(r"\d,\d", line)
        assert _scraped(line) == (105432, 14877)

    def test_no_line_when_the_session_made_no_api_calls(self):
        assert format_token_usage_line(_agent(0, 0)) is None
        assert format_token_usage_line(None) is None


class TestQuietPathsEmitTheLine:
    def test_q_path_exit_summary_carries_it(self, capsys):
        """``-q``: _print_exit_summary is the emitter, so it lands on stdout."""
        cli_obj = cli.HermesCLI.__new__(cli.HermesCLI)
        cli_obj.session_id = "20260928_032703_d57468"
        cli_obj.conversation_history = [{"role": "user", "content": "hi"}]
        cli_obj.agent = _agent(40329, 2899)
        cli_obj._session_db = None
        cli_obj.session_start = datetime.now()

        cli_obj._print_exit_summary(clear_screen=False)

        assert _scraped(capsys.readouterr().out) == (40329, 2899)

    def test_Q_path_emits_it_beside_the_stderr_session_id(self, monkeypatch, capsys):
        """``-Q`` never calls _print_exit_summary. It must still emit the line, next to the
        session_id it already writes to stderr, or turning quiet mode on loses the run's spend.
        """
        monkeypatch.delenv("HERMES_KANBAN_GOAL_MODE", raising=False)
        monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)

        agent = _agent(63579, 11257)
        agent.run_conversation = lambda **kwargs: {"final_response": "ok"}
        agent.session_id = "s-1"
        the_cli = SimpleNamespace(agent=agent, conversation_history=[], session_id="s-1")

        try:
            cli._run_quiet_single_query(the_cli, "hello")
        except SystemExit as exc:
            assert exc.code == 0

        captured = capsys.readouterr()
        # The consumer matches against stdout and stderr combined.
        assert _scraped(captured.out + captured.err) == (63579, 11257)
