# Arduino firmware for the wheelchair drive

These are the Arduino Mega sketches that run the hub motors, read the wheel
encoders, and talk to the ros2_control hardware interface in this package
(`src/wheelchair_interface.cpp`) over `/dev/ttyACM0` at 115200 baud.

They were written in the Arduino IDE sketchbook (`~/Arduino/`) and copied here
on 2026-09-12 so that the repository holds both halves of the drive layer.
The IDE copies remain the ones to open for flashing; keep the two in sync.

| Sketch | Written | Lines | Boot banner | Notes |
|---|---|---|---|---|
| `v3/v3.ino` | 2025-12-06 15:44 | 445 | `Wheelchair Controller v3.0 Ready` | 2 s hardware watchdog (`wdt_enable(WDTO_2S)`), atomic encoder reads, heap-free parsing, 20 Hz control loop, left-encoder sign fix |
| `final_control/final_control.ino` | 2025-12-06 17:27 | 571 | `Wheelchair Controller with PPM Support & Caster Swivel Prevention Ready` | no watchdog, adds 300 ms pivot on direction change (caster swivel prevention), prints a command help banner |

Both share: PID_v1 and Cytron SmartDriveDuo libraries, PPM radio input on pin 20
with a 500 ms timeout and priority over serial, relay on pin 53, encoders on
pins 18/19 (right) and 3/2 (left), wheel radius 0.1524 m, wheelbase 0.565 m,
PPM velocity caps of 1.0 m/s and 1.0 rad/s.

Which sketch is on the board has not been recorded. Read the banner on the
serial port at power-up (`screen /dev/ttyACM0 115200`) and note it here.

Related but unrelated: `wheelchair_nav/arduino/hub_motor_control/` is a bench
sketch with a 0.37 m wheel and is not the wheelchair firmware.
