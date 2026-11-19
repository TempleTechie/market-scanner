name: Market Scanner
on:
  schedule:
    - cron: "37 14 * * 1-5"   # about 8:07 PM IST, Mon-Fri
  workflow_dispatch:
permissions:
  contents: write
jobs:
  scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: pip
      - run: pip install -r requirements.txt
      - run: python scanner.py
        env:
          TELEGRAM_BOT_TOKEN: ${{ secrets.TELEGRAM_BOT_TOKEN }}
          TELEGRAM_CHAT_ID: ${{ secrets.TELEGRAM_CHAT_ID }}
      - name: Save results
        run: |
          git config user.name "scanner-bot"
          git config user.email "bot@users.noreply.github.com"
          git add -A docs nifty500_backup.csv
          git commit -m "Update scan" || echo "nothing to commit"
          git push
