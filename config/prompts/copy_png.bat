
setlocal enableDelayedExpansion
FOR /l %%N in (1,1,%~n1) do (
    set "n=00000%%N"
    set "TEST=!n:~-5!
    echo !TEST!
    copy /y %1 !TEST!.png
)

ren %1 00000.png

