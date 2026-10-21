@echo off
set GRADLE_VERSION=8.13
set CACHE=%USERPROFILE%\.gradle\director-wrapper\%GRADLE_VERSION%
set BIN=%CACHE%\gradle-%GRADLE_VERSION%\bin\gradle.bat
if not exist "%BIN%" (
  if not exist "%CACHE%" mkdir "%CACHE%"
  powershell -NoProfile -ExecutionPolicy Bypass -Command "$u='https://services.gradle.org/distributions/gradle-%GRADLE_VERSION%-bin.zip'; $z='%CACHE%\\gradle-%GRADLE_VERSION%-bin.zip'; Invoke-WebRequest -UseBasicParsing $u -OutFile $z; Expand-Archive -Force $z '%CACHE%'"
)
call "%BIN%" %*
