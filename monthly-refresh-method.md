Yes—you can continue the monthly refresh on the Free plan. The best part is that the preferred Vriddhi workflow barely needs Codex at all.

As of July 18, 2026, OpenAI says Codex is included with ChatGPT Free, although Free has lower usage limits. You sign in with your normal ChatGPT account; no separate Codex account or API key is required. Because availability can change, recheck the official plan page next month. [OpenAI: Using Codex with your ChatGPT plan](https://help.openai.com/en/articles/11369540-codex-in-chatgpt)

## Recommended: use GitHub Actions

This is the simplest and safest monthly procedure—and it does not consume your Codex allowance.

1. Sign in to [GitHub](https://github.com/sixteenpython/vriddhi-core).
2. Open **Actions**.
3. Select **Monthly research candidate**.
4. Click **Run workflow**.
5. Leave `as_of` blank to use the latest complete market date.
6. Leave `allow_high_turnover` disabled.
7. Wait for the workflow to build, validate and test the candidate.
8. Open the automatically generated pull request.
9. Review:

   - Data-through date
   - Unresolved or stale tickers
   - Portfolio picks and drops
   - Turnover
   - Recommendation-status changes
   - Passing checks

10. Merge the PR.
11. Streamlit automatically redeploys from `master`.
12. Confirm the app footer shows the new release and check the Backtest Evidence chart.

These instructions are already permanently stored in [monthly-refresh-runbook.md](C:/Users/ajayv/Documents/jupyter-python/vriddhi-core/docs/monthly-refresh-runbook.md:1).

## If you want to use the Codex desktop app

After returning to Free:

1. Open the ChatGPT/Codex desktop app.
2. Sign out of the Enterprise workspace if necessary.
3. Sign in with your personal ChatGPT account.
4. Select your **Personal/Free workspace**, rather than the former Enterprise workspace.
5. Choose **Open folder** and select:

   `C:\Users\ajayv\Documents\jupyter-python\vriddhi-core`

There is no special “mount” operation. Opening the folder grants Codex access to that local repository. OpenAI’s desktop documentation explicitly supports opening a local folder as the working context. [ChatGPT desktop quickstart](https://learn.chatgpt.com/docs/app)

Your repository, Git history, Python environment and files remain on your computer; changing the ChatGPT subscription does not move or delete them. Enterprise task history may not move to the personal workspace, which is why keeping the operational documentation inside the repository—as we have done—is important.

Then paste this prompt:

> Work in the vriddhi-core repository. Read `docs/monthly-refresh-runbook.md` and `docs/backend-reengineering-reference.md`. Perform the monthly refresh using the latest complete market date. Preserve unrelated working-tree changes. Use the locked uv environment, run the transactional refresh with publication, verify all tests, push only validated generated artifacts to GitHub master, and confirm that the Streamlit app is live with the new release date.

Codex should ultimately execute:

```powershell
uv sync --frozen --all-extras
uv run python vriddhi_monthly_refresh.py --yes --push
```

That single refresh command:

- Creates a backup.
- Builds in isolated staging.
- Downloads current market data.
- Rebuilds all five horizons.
- Validates artifacts and turnover.
- Runs Streamlit smoke tests.
- Promotes only a passing candidate.
- Commits only generated research artifacts.
- Pushes them to GitHub `master`.
- Triggers Streamlit deployment.

## Completely manual fallback

You can also refresh without Codex:

```powershell
cd C:\Users\ajayv\Documents\jupyter-python\vriddhi-core
git pull github master
uv sync --frozen --all-extras
uv run python vriddhi_monthly_refresh.py --yes --push
```

The pipeline is transactional, so if data retrieval, validation or testing fails, the currently published live application remains unchanged.

Before next month, I recommend committing or safely setting aside your currently uncommitted Markdown edits and PDF. A clean working tree will make the monthly operation easier to review.

My recommendation is therefore:

1. **Primary:** one-click GitHub Actions workflow.
2. **Secondary:** Free-plan Codex desktop with the reusable prompt above.
3. **Emergency fallback:** run the three PowerShell commands manually.

The Free-plan limit should not endanger Vriddhi because the actual intelligence and safety of the monthly refresh now live in the repository and CI pipeline—not in this particular Codex conversation.
