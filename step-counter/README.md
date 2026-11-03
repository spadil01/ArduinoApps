# 👣 Step Counter

Sensors on one ankle -> a neural network (trained on the laptop with PyTorch)
counts steps. The browser page and laptop scripts live in AI_Robotics/step_counter.

The app **records trials** for training and **counts steps live** with the
trained network.

## Results page (demo day)

Open **http://<board address>:7000** on a laptop or phone on the same Wi-Fi
(deploy.py prints the address). It shows steps and distance walked, each with
a ± from the fair test, plus a Start over button.

After training again (`python step_counter/train.py` on the laptop copies the
new network into `python/model/`), deploy this app so the board uses it.

## Wiring

| Part | Pin | UNO Q |
|---|---|---|
| MPU-6050 | VCC / GND / SDA / SCL | 3.3V / GND / SDA / SCL |
| HC-SR04 | VCC / GND / Trig | 5V / GND / D7 |
| HC-SR04 | Echo | 1 kΩ resistor -> D8, and 2 kΩ from D8 to GND |

The two resistors shrink the HC-SR04's 5V answer to a safe ~3.3V (no 2 kΩ?
two 1 kΩ in a row work the same).

## Laptop camera (films your feet for the answer key)

Point the laptop's camera (or a USB webcam) at the floor where you walk, then run
`python step_counter/camera.py` on the laptop (`--camera 1` for a USB webcam).
It films every trial on its own; press q in its window to quit.

## Recording a trial

Open AI_Robotics/step_counter/index.html, pick the activity, then press Record:

10 s to get in place -> stand still 2 s -> walk, counting every step out loud ->
stand still 2 s -> Stop -> type in the count.

Trials are saved on the board in `data/`. Copy them to the laptop with
`python step_counter/get_data.py`.
