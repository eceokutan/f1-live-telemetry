Currently only supports assetto corsa

Assetto Corsa Setup Steps:
1) enable shared memory in AC. 
    - Go to Options -> General -> scroll to UI Modules and Shared memory
    - Shared Memory = ON
    - Shared Memory Layout = 1
    - UDP Plugin = OFF
    - UDP Frequency = 333 Hz
    - Enable Data > Apps = ON
2) Start a session on track
    - Telem becomes active when car is loaded, ur in cockpit, phyics thread is running.
    - So, start a Practice/Hotlap/Race session
    - Wait untill car is fully loaded and look around once to ensure physics engine is running.
3) Check if shared memo file exists (only Windows)
    - While AC running press ctrl+shift+esc
    - go to details
    - look for acs.exe / acs_x64.exe
    - right click -> open file location
    the script reads these shared memo files: acpmf_static, acpmf_physics, acpmf_graphic
4) run on the same user session
    - run ac normally.
    - do not run one as admin and one as non-admin
    - run script from the same windows user account.
    
Current notes:
- static info & live labels are not wired yet. (driver, track, etc. missing in ui)
-

TO-DO:
-add assetto corsa competizione