"""ニュース原本の版別取込が再実行と同一版の改変にどう応答するか検証する。"""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from fundamentals.__main__ import main
from fundamentals.news import load_export


class NewsImportCliTests(unittest.TestCase):
    """コマンドの返り値、表示、保存原本をまとめて検証する。"""

    def setUp(self) -> None:
        """外部原本と分離した入力ファイルおよび保存先を用意する。

        引数:
            self (NewsImportCliTests): このテストケース。
        返り値:
            None: なし。
        """
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.input = root / "取得結果.json"
        self.archive = root / "版別原本"
        self.article = {
            "id": "記事一", "revision_id": "初版", "headline": "国内材料の公表",
            "body": "当初版として取得された本文。", "canonical_url": "https://example.com/news/1",
            "published_at": "2026-10-15T07:00:00+09:00", "fetched_at": "2026-10-15T07:01:00+09:00",
        }
        self.input.write_text(json.dumps({"article": self.article}, ensure_ascii=False), encoding="utf-8")

    def _import(self) -> tuple[int, str, str]:
        """実際のコマンド引数で取込処理を実行し、表示を捕捉する。

        引数:
            self (NewsImportCliTests): このテストケース。
        返り値:
            tuple[int, str, str]: 終了コード、標準出力、標準エラー。
        """
        output, errors = io.StringIO(), io.StringIO()
        arguments = ["fundamentals", "import-news", "--input", str(self.input), "--archive", str(self.archive)]
        with patch("sys.argv", arguments), redirect_stdout(output), redirect_stderr(errors):
            code = main()
        return code, output.getvalue(), errors.getvalue()

    # ----------------------------------------

    def test_reimporting_same_original_saves_zero_new_revisions(self) -> None:
        """同じ原本を二度取り込んでも増殖させず、新規保存をゼロと報告する。

        引数:
            self (NewsImportCliTests): このテストケース。
        返り値:
            None: なし。
        """
        first_code, first_output, first_errors = self._import()
        files = list(self.archive.glob("*.json"))
        self.assertEqual(len(files), 1)
        original_bytes = files[0].read_bytes()
        second_code, second_output, second_errors = self._import()
        self.assertEqual((first_code, second_code), (0, 0))
        self.assertEqual((first_errors, second_errors), ("", ""))
        self.assertEqual(json.loads(first_output)["新規保存件数"], 1)
        self.assertEqual(json.loads(second_output)["入力件数"], 1)
        self.assertEqual(json.loads(second_output)["新規保存件数"], 0)
        self.assertEqual(list(self.archive.glob("*.json")), files)
        self.assertEqual(files[0].read_bytes(), original_bytes)
        self.assertEqual(load_export(self.archive), [self.article])

    def test_changed_body_with_same_revision_id_is_rejected_without_overwrite(self) -> None:
        """同じ記事版の異なる本文を拒否し、当初の保存内容を保持する。

        引数:
            self (NewsImportCliTests): このテストケース。
        返り値:
            None: なし。
        """
        self.assertEqual(self._import()[0], 0)
        original_path = next(self.archive.glob("*.json"))
        original_bytes = original_path.read_bytes()
        changed = {**self.article, "body": "同じ版IDのまま差し替えられた本文。"}
        self.input.write_text(json.dumps({"article": changed}, ensure_ascii=False), encoding="utf-8")
        code, output, errors = self._import()
        self.assertEqual(code, 1)
        self.assertEqual(output, "")
        self.assertIn("同じ記事版IDの本文が変化", errors)
        self.assertEqual(original_path.read_bytes(), original_bytes)
        self.assertEqual(load_export(self.archive), [self.article])
        self.assertEqual(len(list(self.archive.glob("*.json"))), 1)
