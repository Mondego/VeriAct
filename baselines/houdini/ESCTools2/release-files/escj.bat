@echo off

rem To install ESC/Java2 on a new machine, change the following two
rem lines appropriately:
set ESCJAVA_ROOT=C:\ESCJava2
set JAVA=java

rem Some arguments to Simplify, passed via environment variables
rem set ESCJ_SIMPLIFY=%ESCJAVA_ROOT%\bin\Simplify-1.5.4.exe
set ESCJ_SIMPLIFY=%ESCJAVA_ROOT%\Simplify-1.5.4.exe
set ESCJ_SIMPLIFY_ARGS=-noprune -noplunge
set PROVER_KILL_TIME=300
set PROVER_CC_LIMIT=10
set PROVER_MAX_FNUR=700

if "%ESCJ_STDARGS%"=="" set ESCJ_STDARGS=-nowarn Deadlock -specs %ESCJAVA_ROOT%\specs

rem ESCJ_ARGS is a variable local to this batch file
set ESCJ_ARGS=%1
:getargs
shift
if "%1"=="" goto endGetargs
set ESCJ_ARGS=%ESCJ_ARGS% %1
goto getargs
:endGetargs

rem USERPATH is a variable local to this batch file
if "%CLASSPATH%"=="" set USERPATH=.
if not "%CLASSPATH%"=="" set USERPATH=%CLASSPATH%
@echo on
"%JAVA%"  -Dsimplify=%ESCJ_SIMPLIFY% -classpath "%ESCJAVA_ROOT%\esctools2.jar;%ESCJAVA_ROOT%\Utils\BCEL\bcel-5.2\bcel-5.2.jar;%ESCJAVA_ROOT%\Utils\Javafe\Javafe2.0.5b.jar" escjava.Main -classpath   %ESCJAVA_ROOT%\jmlspecs.jar -classpath . %ESCJ_STDARGS% %ESCJ_ARGS%