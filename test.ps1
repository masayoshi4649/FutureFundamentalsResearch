<#
機能: 先物基準と情報時点の回帰テストを実行する。
引数: なし。
返り値: 終了コード（整数）。成功０、失敗時はテスト終了コード。
#>
$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    python -X utf8 -m unittest discover -s tests -v
    $testExitCode = $LASTEXITCODE
}
finally {
    Pop-Location
}
exit $testExitCode
