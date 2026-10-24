[CmdletBinding()]
param(
    [string]$InstallRoot = "$env:USERPROFILE\SierraVisionServer",
    [switch]$StartOnly,
    [switch]$NoAutoStart,
    [switch]$SkipAdminBootstrap
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$RepoOwner = 'SAUNAK359'
$RepoName = 'Myprojects'
$Branch = 'sierra-vision-server-v1'
$MinimumWindowsBuild = 22631 # Current Docker Desktop support floor: Windows 11 23H2.
$MinimumRamGb = 8
$RequiredFreeDiskGb = 12
$HealthUrl = 'http://127.0.0.1:8080/health/ready'
$TaskName = 'Sierra Vision Server'
$DockerInstallerUrl = 'https://desktop.docker.com/win/main/amd64/Docker%20Desktop%20Installer.exe'

function Write-Step([string]$Message) {
    Write-Host "`n==> $Message" -ForegroundColor Cyan
}

function Write-Ok([string]$Message) {
    Write-Host "[OK] $Message" -ForegroundColor Green
}

function Write-Warn([string]$Message) {
    Write-Host "[WARN] $Message" -ForegroundColor Yellow
}

function Fail([string]$Message) {
    throw $Message
}

function Test-IsAdministrator {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = [Security.Principal.WindowsPrincipal]::new($identity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Ensure-Elevated {
    if (Test-IsAdministrator) { return }
    if (-not $PSCommandPath) {
        Fail 'Run this script from a saved .ps1 file so it can request administrator access when needed.'
    }

    Write-Host 'Administrator permission is required once to verify/enable WSL 2.' -ForegroundColor Yellow
    $args = @(
        '-NoProfile',
        '-ExecutionPolicy', 'Bypass',
        '-File', "`"$PSCommandPath`"",
        '-InstallRoot', "`"$InstallRoot`""
    )
    if ($NoAutoStart) { $args += '-NoAutoStart' }
    if ($SkipAdminBootstrap) { $args += '-SkipAdminBootstrap' }
    Start-Process -FilePath 'powershell.exe' -Verb RunAs -ArgumentList $args
    exit 0
}

function Add-DockerToCurrentPath {
    $paths = @(
        "$env:LOCALAPPDATA\Programs\DockerDesktop\resources\bin",
        "$env:ProgramFiles\Docker\Docker\resources\bin"
    )
    foreach ($path in $paths) {
        if ((Test-Path $path) -and (($env:PATH -split ';') -notcontains $path)) {
            $env:PATH = "$path;$env:PATH"
        }
    }
}

function Get-DockerDesktopExe {
    $candidates = @(
        "$env:LOCALAPPDATA\Programs\DockerDesktop\Docker Desktop.exe",
        "$env:ProgramFiles\Docker\Docker\Docker Desktop.exe"
    )
    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) { return $candidate }
    }
    return $null
}

function Test-DockerEngine {
    Add-DockerToCurrentPath
    if (-not (Get-Command docker.exe -ErrorAction SilentlyContinue)) { return $false }
    try {
        docker info --format '{{.ServerVersion}}' 2>$null | Out-Null
        return ($LASTEXITCODE -eq 0)
    } catch {
        return $false
    }
}

function Wait-DockerEngine([int]$TimeoutSeconds = 240) {
    $desktop = Get-DockerDesktopExe
    if (-not (Test-DockerEngine)) {
        if (-not $desktop) { return $false }
        Write-Step 'Starting Docker Desktop'
        Start-Process -FilePath $desktop | Out-Null
    }

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        if (Test-DockerEngine) { return $true }
        Start-Sleep -Seconds 3
    }
    return $false
}

function Get-EnvValue([string]$Path, [string]$Name) {
    if (-not (Test-Path $Path)) { return $null }
    $line = Get-Content $Path | Where-Object { $_ -match "^$([regex]::Escape($Name))=" } | Select-Object -Last 1
    if (-not $line) { return $null }
    return $line.Substring($Name.Length + 1)
}

function Set-EnvValue([string]$Content, [string]$Name, [string]$Value) {
    $pattern = "(?m)^$([regex]::Escape($Name))=.*$"
    $replacement = "$Name=$Value"
    if ([regex]::IsMatch($Content, $pattern)) {
        return [regex]::Replace($Content, $pattern, [System.Text.RegularExpressions.MatchEvaluator]{ param($m) $replacement })
    }
    return $Content.TrimEnd() + "`r`n$replacement`r`n"
}

function New-RandomBytes([int]$Count) {
    $bytes = New-Object byte[] $Count
    [Security.Cryptography.RandomNumberGenerator]::Fill($bytes)
    return $bytes
}

function ConvertTo-Base64Url([byte[]]$Bytes, [bool]$KeepPadding = $false) {
    $value = [Convert]::ToBase64String($Bytes).Replace('+', '-').Replace('/', '_')
    if (-not $KeepPadding) { $value = $value.TrimEnd('=') }
    return $value
}

function Protect-SecretFile([string]$Path) {
    try {
        $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
        $acl = [Security.AccessControl.FileSecurity]::new()
        $acl.SetOwner($identity.User)
        $acl.SetAccessRuleProtection($true, $false)
        $rule = [Security.AccessControl.FileSystemAccessRule]::new(
            $identity.User,
            [Security.AccessControl.FileSystemRights]::FullControl,
            [Security.AccessControl.AccessControlType]::Allow
        )
        $acl.AddAccessRule($rule)
        Set-Acl -Path $Path -AclObject $acl
        Write-Ok 'Restricted .env permissions to the current Windows account.'
    } catch {
        Write-Warn "Could not tighten .env ACL automatically: $($_.Exception.Message)"
    }
}

function Test-Port8080Available {
    try {
        $listeners = Get-NetTCPConnection -State Listen -LocalPort 8080 -ErrorAction SilentlyContinue
        if ($listeners) {
            $dockerOwns = $false
            foreach ($listener in $listeners) {
                try {
                    $process = Get-Process -Id $listener.OwningProcess -ErrorAction Stop
                    if ($process.ProcessName -match 'docker|com\.docker') { $dockerOwns = $true }
                } catch { }
            }
            if (-not $dockerOwns) {
                Fail 'TCP port 8080 is already in use by another application. Stop that application before installing Sierra Server.'
            }
        }
    } catch {
        if ($_.Exception.Message -like '*already in use*') { throw }
        Write-Warn 'Could not inspect port 8080 with Get-NetTCPConnection; Docker binding will be verified after startup.'
    }
}

function Ensure-WindowsPrerequisites {
    Write-Step 'Checking Windows, memory, virtualization and disk'

    if (-not [Environment]::Is64BitOperatingSystem) {
        Fail 'Sierra Server requires 64-bit Windows.'
    }

    $currentVersion = Get-ItemProperty 'HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion'
    $build = [int]$currentVersion.CurrentBuildNumber
    Write-Host "Windows: $($currentVersion.ProductName) build $build"
    if ($build -lt $MinimumWindowsBuild) {
        Write-Warn "Current Docker Desktop requires Windows 11 23H2 / build $MinimumWindowsBuild or newer."
        Start-Process 'ms-settings:windowsupdate' | Out-Null
        Fail 'Update Windows, reboot, then run this installer again.'
    }

    $computer = Get-CimInstance Win32_ComputerSystem
    $ramGb = [math]::Round($computer.TotalPhysicalMemory / 1GB, 1)
    Write-Host "RAM: $ramGb GB"
    if ($ramGb -lt $MinimumRamGb) {
        Fail "Docker Desktop requires at least $MinimumRamGb GB RAM for this deployment."
    }

    $processor = Get-CimInstance Win32_Processor | Select-Object -First 1
    $virtualizationReady = ($processor.VirtualizationFirmwareEnabled -eq $true) -or ($computer.HypervisorPresent -eq $true)
    if (-not $virtualizationReady) {
        Fail 'CPU virtualization is disabled. Enable Intel VT-x/AMD-V in BIOS/UEFI, reboot, then rerun the installer.'
    }
    Write-Ok 'Hardware virtualization is enabled.'

    $rootPath = [IO.Path]::GetPathRoot([IO.Path]::GetFullPath($InstallRoot))
    $driveName = $rootPath.TrimEnd('\').TrimEnd(':')
    $drive = Get-PSDrive -Name $driveName
    $freeGb = [math]::Round($drive.Free / 1GB, 1)
    Write-Host "Free disk on ${rootPath}: $freeGb GB"
    if ($freeGb -lt $RequiredFreeDiskGb) {
        Fail "At least $RequiredFreeDiskGb GB free disk space is required before setup."
    }
}

function Ensure-Wsl2 {
    Write-Step 'Checking WSL 2'
    $restartRequired = $false

    $features = @('Microsoft-Windows-Subsystem-Linux', 'VirtualMachinePlatform')
    foreach ($featureName in $features) {
        $feature = Get-WindowsOptionalFeature -Online -FeatureName $featureName
        if ($feature.State -ne 'Enabled') {
            Write-Host "Enabling Windows feature: $featureName"
            $result = Enable-WindowsOptionalFeature -Online -FeatureName $featureName -All -NoRestart
            if ($result.RestartNeeded) { $restartRequired = $true }
        }
    }

    if ($restartRequired) {
        Write-Warn 'WSL 2 Windows features were enabled and Windows must restart before Docker can be installed.'
        Write-Host "After reboot, run the same installer again: $PSCommandPath" -ForegroundColor Yellow
        exit 3010
    }

    try {
        & wsl.exe --update | Out-Host
    } catch {
        Write-Warn "WSL update returned an error: $($_.Exception.Message)"
    }

    try { & wsl.exe --set-default-version 2 | Out-Null } catch { }
    $versionText = (& wsl.exe --version 2>&1 | Out-String).Trim()
    if (-not $versionText) {
        Fail 'WSL is enabled but its modern version could not be verified. Run "wsl --update", reboot if requested, then rerun.'
    }
    Write-Host $versionText

    $match = [regex]::Match($versionText, '(?m)\b(\d+\.\d+\.\d+(?:\.\d+)?)\b')
    if ($match.Success) {
        try {
            $version = [version]$match.Groups[1].Value
            if ($version -lt [version]'2.1.5') {
                Fail 'WSL 2.1.5 or newer is required. Run "wsl --update" and rerun this installer.'
            }
        } catch { }
    }
    Write-Ok 'WSL 2 is available.'
}

function Install-DockerDesktopIfNeeded {
    Add-DockerToCurrentPath
    if ((Get-Command docker.exe -ErrorAction SilentlyContinue) -and (Get-DockerDesktopExe)) {
        Write-Ok 'Docker Desktop is already installed.'
        return
    }

    Write-Step 'Docker Desktop is missing'
    Write-Host 'Sierra uses Docker Desktop with the WSL 2 Linux backend.'
    Write-Host 'Docker Desktop is free for personal use and qualifying small businesses; other commercial use may require a paid Docker subscription.'
    $answer = Read-Host 'Download Docker Desktop from docker.com and accept its license during installation? [y/N]'
    if ($answer -notmatch '^(y|yes)$') {
        Fail 'Docker Desktop is required. Install it manually, then rerun this script.'
    }

    $installer = Join-Path $env:TEMP 'Sierra-Docker-Desktop-Installer.exe'
    Write-Host 'Downloading official Docker Desktop installer...'
    Invoke-WebRequest -Uri $DockerInstallerUrl -OutFile $installer -UseBasicParsing

    $signature = Get-AuthenticodeSignature $installer
    if ($signature.Status -ne 'Valid' -or $signature.SignerCertificate.Subject -notmatch 'Docker') {
        Remove-Item $installer -Force -ErrorAction SilentlyContinue
        Fail 'Docker installer signature validation failed. The installer was deleted.'
    }
    Write-Ok "Docker installer signature verified: $($signature.SignerCertificate.Subject)"

    Write-Host 'Installing Docker Desktop in recommended per-user mode with WSL 2...'
    $process = Start-Process -FilePath $installer -Wait -PassThru -ArgumentList @(
        'install', '--user', '--quiet', '--accept-license', '--backend=wsl-2'
    )
    Remove-Item $installer -Force -ErrorAction SilentlyContinue
    if ($process.ExitCode -ne 0) {
        Fail "Docker Desktop installer exited with code $($process.ExitCode)."
    }

    Add-DockerToCurrentPath
    if (-not (Get-DockerDesktopExe)) {
        Fail 'Docker Desktop installation completed but its executable was not found.'
    }
    Write-Ok 'Docker Desktop installed.'
}

function Ensure-DockerReady {
    Write-Step 'Checking Docker engine and Compose'
    if (-not (Wait-DockerEngine 240)) {
        Fail 'Docker Desktop did not become ready. Open Docker Desktop, confirm it starts successfully, then rerun this script.'
    }

    $osType = (docker info --format '{{.OSType}}').Trim()
    if ($osType -ne 'linux') {
        Fail 'Docker is currently using Windows containers. Switch Docker Desktop to Linux containers and rerun.'
    }

    docker compose version | Out-Host
    if ($LASTEXITCODE -ne 0) { Fail 'Docker Compose v2 is not available.' }
    Write-Ok 'Docker engine is ready in Linux-container mode.'
}

function Get-LatestServerSource([string]$Destination) {
    Write-Step 'Downloading the exact Sierra Server branch commit from GitHub'
    $headers = @{ 'User-Agent' = 'SierraVisionServerInstaller/1.0' }
    $commit = Invoke-RestMethod -Headers $headers -Uri "https://api.github.com/repos/$RepoOwner/$RepoName/commits/$Branch"
    $sha = [string]$commit.sha
    if ($sha -notmatch '^[0-9a-f]{40}$') { Fail 'GitHub did not return a valid source commit SHA.' }

    $zip = Join-Path $env:TEMP "SierraServer-$sha.zip"
    $extract = Join-Path $env:TEMP "SierraServer-$sha-extract"
    Remove-Item $zip -Force -ErrorAction SilentlyContinue
    Remove-Item $extract -Recurse -Force -ErrorAction SilentlyContinue

    Invoke-WebRequest -Uri "https://github.com/$RepoOwner/$RepoName/archive/$sha.zip" -OutFile $zip -UseBasicParsing
    Expand-Archive -Path $zip -DestinationPath $extract -Force

    $repoRoot = Get-ChildItem $extract -Directory | Select-Object -First 1
    if (-not $repoRoot) { Fail 'Downloaded repository archive was empty.' }
    $serverSource = Join-Path $repoRoot.FullName 'SierraServer'
    foreach ($required in @('docker-compose.yml', 'Dockerfile', '.env.example', 'app\main.py', 'scripts\bootstrap_admin.py')) {
        if (-not (Test-Path (Join-Path $serverSource $required))) {
            Fail "Downloaded server source is missing required file: $required"
        }
    }

    if (Test-Path $Destination) {
        Write-Host 'Existing Sierra Server installation found; stopping containers before updating.'
        if (Test-Path (Join-Path $Destination 'docker-compose.yml')) {
            Push-Location $Destination
            try { docker compose down --remove-orphans | Out-Host } catch { }
            Pop-Location
        }
    }

    $envBackup = Join-Path $env:TEMP "sierra-env-$([Guid]::NewGuid().ToString('N')).backup"
    $oldEnv = Join-Path $Destination '.env'
    if (Test-Path $oldEnv) { Copy-Item $oldEnv $envBackup -Force }

    if (Test-Path $Destination) { Remove-Item $Destination -Recurse -Force }
    New-Item -ItemType Directory -Path (Split-Path -Parent $Destination) -Force | Out-Null
    Move-Item -Path $serverSource -Destination $Destination

    if (Test-Path $envBackup) {
        Copy-Item $envBackup (Join-Path $Destination '.env') -Force
        Remove-Item $envBackup -Force
        Write-Ok 'Preserved the existing local .env secrets.'
    }

    [IO.File]::WriteAllText((Join-Path $Destination '.sierra-source-commit'), "$sha`n", [Text.UTF8Encoding]::new($false))
    Remove-Item $zip -Force -ErrorAction SilentlyContinue
    Remove-Item $extract -Recurse -Force -ErrorAction SilentlyContinue
    Write-Ok "Installed Sierra Server source commit $sha"
    return $sha
}

function Ensure-LocalEnvironment([string]$ServerRoot) {
    Write-Step 'Checking local Sierra secrets/configuration'
    $envFile = Join-Path $ServerRoot '.env'
    $example = Join-Path $ServerRoot '.env.example'

    if (-not (Test-Path $envFile)) {
        $volumeExists = $false
        try {
            docker volume inspect 'sierra-vision_sierra_pgdata' *> $null
            $volumeExists = ($LASTEXITCODE -eq 0)
        } catch { }
        if ($volumeExists) {
            Fail 'The Sierra PostgreSQL volume exists but .env is missing. Restore the original .env/database password before continuing.'
        }

        $content = Get-Content $example -Raw
        $dbPassword = ConvertTo-Base64Url (New-RandomBytes 32)
        $jwtSecret = ConvertTo-Base64Url (New-RandomBytes 48)
        $providerKey = ConvertTo-Base64Url (New-RandomBytes 32) $true

        $content = Set-EnvValue $content 'COMPOSE_PROJECT_NAME' 'sierra-vision'
        $content = Set-EnvValue $content 'SIERRA_ENV' 'development'
        $content = Set-EnvValue $content 'SIERRA_DB_PASSWORD' $dbPassword
        $content = Set-EnvValue $content 'SIERRA_DATABASE_URL' "postgresql+asyncpg://sierra:$dbPassword@db:5432/sierra"
        $content = Set-EnvValue $content 'SIERRA_JWT_SECRET' $jwtSecret
        $content = Set-EnvValue $content 'SIERRA_PROVIDER_MASTER_KEY' $providerKey
        $content = Set-EnvValue $content 'SIERRA_PUBLIC_BASE_URL' ''
        $content = Set-EnvValue $content 'SIERRA_CLOUDFLARE_TUNNEL_TOKEN' ''

        [IO.File]::WriteAllText($envFile, $content, [Text.UTF8Encoding]::new($false))
        Protect-SecretFile $envFile
        Write-Ok 'Generated new database/JWT/provider-encryption secrets locally. They were not printed.'
    } else {
        foreach ($name in @('SIERRA_DB_PASSWORD', 'SIERRA_JWT_SECRET', 'SIERRA_PROVIDER_MASTER_KEY')) {
            $value = Get-EnvValue $envFile $name
            if (-not $value -or $value -match 'CHANGE_ME|change_me') {
                Fail "$name is missing or still contains a placeholder in .env. Refusing to start with unsafe secrets."
            }
        }
        $content = Get-Content $envFile -Raw
        if (-not (Get-EnvValue $envFile 'COMPOSE_PROJECT_NAME')) {
            $content = Set-EnvValue $content 'COMPOSE_PROJECT_NAME' 'sierra-vision'
            [IO.File]::WriteAllText($envFile, $content, [Text.UTF8Encoding]::new($false))
        }
        Protect-SecretFile $envFile
        Write-Ok 'Existing local secrets passed placeholder checks.'
    }
}

function Start-SierraLocal([string]$ServerRoot) {
    Ensure-DockerReady
    Test-Port8080Available

    Write-Step 'Starting Sierra PostgreSQL and API locally'
    Push-Location $ServerRoot
    try {
        try { docker compose --profile tunnel --profile public stop cloudflared caddy 2>$null | Out-Null } catch { }
        docker compose pull db | Out-Host
        if ($LASTEXITCODE -ne 0) { Fail 'Could not download the PostgreSQL image.' }
        docker compose build --pull api | Out-Host
        if ($LASTEXITCODE -ne 0) { Fail 'Sierra API image build failed.' }
        docker compose up -d db api | Out-Host
        if ($LASTEXITCODE -ne 0) { Fail 'Could not start Sierra API/database containers.' }

        Write-Host 'Waiting for health endpoint...'
        $deadline = (Get-Date).AddMinutes(3)
        $healthy = $false
        while ((Get-Date) -lt $deadline) {
            try {
                $health = Invoke-RestMethod -Uri $HealthUrl -TimeoutSec 4
                if ($health.ok -eq $true -and $health.database -eq $true) {
                    $healthy = $true
                    break
                }
            } catch { }
            Start-Sleep -Seconds 3
        }
        if (-not $healthy) {
            docker compose logs --tail=160 api db | Out-Host
            Fail 'Sierra API did not become healthy.'
        }

        $published = (docker compose port api 8080 | Select-Object -First 1).Trim()
        if ($published -notmatch '^127\.0\.0\.1:8080$') {
            Fail "Unsafe/unexpected API binding detected: '$published'. Expected 127.0.0.1:8080."
        }
        Write-Ok 'API is bound to localhost only.'

        docker compose exec -T api python -m pip check | Out-Host
        if ($LASTEXITCODE -ne 0) { Fail 'Python dependency consistency check failed inside the API container.' }

        docker compose exec -T api alembic current | Out-Host
        if ($LASTEXITCODE -ne 0) { Fail 'Database migration verification failed.' }

        docker compose exec -T api python -c "from app.config import settings; assert len(settings.jwt_secret.get_secret_value()) >= 32; from app.provider_crypto import encrypt_secret,decrypt_secret; x=encrypt_secret('sierra-self-test'); assert decrypt_secret(x)=='sierra-self-test'; print('security-config: OK')" | Out-Host
        if ($LASTEXITCODE -ne 0) { Fail 'Server cryptographic configuration self-test failed.' }

        Write-Ok 'Sierra API, database, migrations and cryptographic configuration are healthy.'
    } finally {
        Pop-Location
    }
}

function Ensure-Admin([string]$ServerRoot) {
    if ($SkipAdminBootstrap) {
        Write-Warn 'Admin bootstrap was skipped by request.'
        return
    }

    Push-Location $ServerRoot
    try {
        $countRaw = docker compose exec -T db psql -U sierra -d sierra -tAc "SELECT COUNT(*) FROM users WHERE role='admin';"
        if ($LASTEXITCODE -ne 0) { Fail 'Could not query the Sierra database for an existing administrator.' }
        $count = 0
        [void][int]::TryParse(($countRaw | Out-String).Trim(), [ref]$count)
        if ($count -gt 0) {
            Write-Ok "Administrator account already exists ($count found); bootstrap skipped."
            return
        }

        Write-Step 'Creating the first Sierra administrator securely'
        Write-Host 'Your password is typed directly into the temporary API container, hashed immediately, and is never written to .env.' -ForegroundColor Yellow
        docker compose run --rm api python scripts/bootstrap_admin.py
        if ($LASTEXITCODE -ne 0) { Fail 'Administrator bootstrap failed.' }

        $verifyRaw = docker compose exec -T db psql -U sierra -d sierra -tAc "SELECT COUNT(*) FROM users WHERE role='admin';"
        $verify = 0
        [void][int]::TryParse(($verifyRaw | Out-String).Trim(), [ref]$verify)
        if ($verify -lt 1) { Fail 'Administrator bootstrap returned success but no admin account exists.' }
        Write-Ok 'Administrator account verified in PostgreSQL.'
    } finally {
        Pop-Location
    }
}

function Register-AutoStart([string]$ServerRoot) {
    if ($NoAutoStart) {
        Write-Warn 'Windows sign-in auto-start was disabled by -NoAutoStart.'
        return
    }

    $installedScript = Join-Path $ServerRoot 'scripts\Install-SierraVisionServer.ps1'
    if (-not (Test-Path $installedScript)) {
        Write-Warn 'Installed startup script is missing; automatic startup was not registered.'
        return
    }

    try {
        $currentUser = [Security.Principal.WindowsIdentity]::GetCurrent().Name
        $action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument (
            "-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$installedScript`" -StartOnly -InstallRoot `"$ServerRoot`""
        )
        $trigger = New-ScheduledTaskTrigger -AtLogOn -User $currentUser
        $settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit ([TimeSpan]::Zero)
        Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings -Description 'Starts local Sierra Vision Server containers after Windows sign-in.' -Force | Out-Null
        Write-Ok 'Registered Sierra Server to start automatically after Windows sign-in.'
    } catch {
        Write-Warn "Could not register Windows auto-start task: $($_.Exception.Message)"
    }
}

function Show-FinalStatus([string]$ServerRoot, [string]$SourceCommit) {
    Write-Step 'Final local deployment verification'
    Push-Location $ServerRoot
    try {
        docker compose ps | Out-Host
        $health = Invoke-RestMethod -Uri $HealthUrl -TimeoutSec 5
        Write-Host "Health: ok=$($health.ok), database=$($health.database), cloud_ready=$($health.cloud_ready)"
    } finally {
        Pop-Location
    }

    Write-Host ''
    Write-Host '============================================================' -ForegroundColor Green
    Write-Host '  SIERRA VISION SERVER - LOCAL SETUP COMPLETE' -ForegroundColor Green
    Write-Host '============================================================' -ForegroundColor Green
    Write-Host "Installed source: $SourceCommit"
    Write-Host "Server folder:    $ServerRoot"
    Write-Host 'Health:           http://127.0.0.1:8080/health/ready'
    Write-Host 'API docs (local): http://127.0.0.1:8080/docs'
    Write-Host 'Admin:            http://127.0.0.1:8080/admin/login'
    Write-Host ''
    Write-Host 'PUBLIC INTERNET ACCESS IS STILL OFF.' -ForegroundColor Yellow
    Write-Host 'Do not port-forward 8080. We will configure the Cloudflare HTTPS tunnel only after local testing.' -ForegroundColor Yellow
    Write-Host ''
    Write-Host 'Useful commands:'
    Write-Host "  cd `"$ServerRoot`""
    Write-Host '  docker compose ps'
    Write-Host '  docker compose logs -f api'
    Write-Host '  docker compose restart api'
    Write-Host '  docker compose down              # keeps PostgreSQL volume/data'
}

function Start-OnlyMode {
    Add-DockerToCurrentPath
    if (-not (Wait-DockerEngine 240)) { exit 1 }
    if (-not (Test-Path (Join-Path $InstallRoot 'docker-compose.yml'))) { exit 2 }
    Push-Location $InstallRoot
    try {
        docker compose up -d db api *> $null
        $deadline = (Get-Date).AddSeconds(90)
        while ((Get-Date) -lt $deadline) {
            try {
                $health = Invoke-RestMethod -Uri $HealthUrl -TimeoutSec 3
                if ($health.ok -eq $true) { exit 0 }
            } catch { }
            Start-Sleep -Seconds 3
        }
        exit 3
    } finally {
        Pop-Location
    }
}

try {
    if ($StartOnly) {
        Start-OnlyMode
    }

    if ($PSVersionTable.PSVersion.Major -lt 5) {
        Fail 'PowerShell 5.1 or newer is required.'
    }

    Write-Host ''
    Write-Host '============================================================' -ForegroundColor Yellow
    Write-Host '  Sierra Vision Server - Secure Windows Installer' -ForegroundColor Yellow
    Write-Host '============================================================' -ForegroundColor Yellow
    Write-Host 'This installs LOCAL-ONLY first. It does not open router ports or enable the public tunnel.'

    Ensure-Elevated
    Ensure-WindowsPrerequisites
    Ensure-Wsl2
    Install-DockerDesktopIfNeeded
    Ensure-DockerReady

    $sourceCommit = Get-LatestServerSource $InstallRoot
    Ensure-LocalEnvironment $InstallRoot
    Start-SierraLocal $InstallRoot
    Ensure-Admin $InstallRoot
    Register-AutoStart $InstallRoot
    Show-FinalStatus $InstallRoot $sourceCommit
} catch {
    Write-Host ''
    Write-Host 'SIERRA SETUP STOPPED' -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    Write-Host ''
    Write-Host 'Nothing should be exposed publicly. Fix the item above and rerun the same script.' -ForegroundColor Yellow
    exit 1
}
