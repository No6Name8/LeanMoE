# PowerShell 5.1 can turn native stderr (including successful Caddy diagnostics)
# into a terminating error when the caller redirects streams with Stop enabled.
# Preserve stderr privately and judge native commands by their actual exit code.
function Invoke-GuestTool {
    param([string]$Executable, [string[]]$ToolArguments, [string]$LogName)
    if ($LogName -notmatch '^[a-z-]+$') { throw 'Invalid private log name.' }
    $base = 'C:\LeanMoE-private\' + $LogName
    $process = Start-Process -FilePath $Executable -ArgumentList $ToolArguments -WindowStyle Hidden -Wait -PassThru -RedirectStandardOutput ($base+'.stdout.log') -RedirectStandardError ($base+'.stderr.log')
    if ($process.ExitCode -ne 0) { throw "Guest tool failed ($LogName, exit $($process.ExitCode)); inspect private logs." }
}
