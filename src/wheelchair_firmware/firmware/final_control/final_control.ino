#include <PID_v1.h>
#include <Cytron_SmartDriveDuo.h>

// ================== PPM CONFIG ==================
#define PPM_PIN 20          // PPM signal input pin
#define CHANNELS 8         // Number of PPM channels

volatile uint16_t ppmValues[CHANNELS];
volatile uint8_t ppmIndex = 0;
volatile unsigned long lastMicros = 0;

// PPM variables
int ch1, ch2;
bool ppm_active = false;
unsigned long last_ppm_time = 0;
const unsigned long PPM_TIMEOUT = 500; // 500ms timeout for PPM signal

// Encoder pins
int encoderPin1R = 18;
int encoderPin2R = 19;
int encoderPin1L = 3;
int encoderPin2L = 2;

// Relay Pin
int relayPin = 53;

// Motor driver pins
#define IN1 7
#define AN1 5
#define AN2 4
#define IN2 6

Cytron_SmartDriveDuo smartDriveDuo(PWM_INDEPENDENT, IN1, IN2, AN1, AN2);

// Encoder variables
volatile int lastEncodedL = 0;
volatile long encoderValueL = 0;
volatile int lastEncodedR = 0;
volatile long encoderValueR = 0;

// Wheel specifications
const long CPR = 10000;
const float WHEEL_RADIUS = 0.1524; // 6 inches
const float WHEELBASE = 0.565; // 57 cm
const float MAX_ANGULAR_VEL = 17.59; // rad/s
const float MAX_LINEAR_VEL = 2.68; // m/s (WHEEL_RADIUS * MAX_ANGULAR_VEL)

// PPM velocity limits (can be different from serial limits for safety)
const float PPM_MAX_LINEAR_VEL = 1.0;   // 1 m/s max for PPM
const float PPM_MAX_ANGULAR_VEL = 1.0;  // 1 rad/s max for PPM

// Timing
unsigned long last_millis = 0;
const unsigned long interval = 50;
long lastEncoderL = 0;
long lastEncoderR = 0;

// CASTER SWIVEL PREVENTION
bool pivot_mode = false;
unsigned long pivot_start_time = 0;
const unsigned long PIVOT_DURATION = 300; // 300ms pivot duration
const float PIVOT_ANGULAR_VEL = 2.0; // rad/s for pivot turn
float prev_linear_velocity = 0.0;
float stored_linear_cmd = 0.0;
float stored_angular_cmd = 0.0;

// TRIPLE COMMAND PARSING SYSTEM
enum CommandMode {
  WHEEL_MODE,    // rp5.0,lp3.0,
  CMDVEL_MODE,   // x:0.5,t:0.1,
  PPM_MODE       // PPM transmitter input
};

CommandMode current_mode = WHEEL_MODE;

// Command parsing variables
bool is_right_wheel_cmd = false;
bool is_left_wheel_cmd = false;
bool is_linear_cmd = false;
bool is_angular_cmd = false;
bool is_right_wheel_forward = true;
bool is_left_wheel_forward = true;
char value[] = "00.00";
uint8_t value_idx = 0;
String right_wheel_sign = "p";
String left_wheel_sign = "p";

// Velocity variables
double right_wheel_cmd_vel = 0.0;
double left_wheel_cmd_vel = 0.0;
double right_wheel_target = 0.0;
double left_wheel_target = 0.0;
double right_wheel_meas_vel = 0.0;
double left_wheel_meas_vel = 0.0;
double right_wheel_cmd = 0.0;
double left_wheel_cmd = 0.0;

// cmd_vel variables
double linear_x = 0.0;
double angular_z = 0.0;

// Acceleration limiting
const double MAX_ACCEL = 5.0;
double filtered_right_vel = 0.0;
double filtered_left_vel = 0.0;
const double FILTER_ALPHA = 0.8;

// PID gains
double Kp_r = 7.0, Ki_r = 8.0, Kd_r = 0.15;
double Kp_l = 7.2, Ki_l = 8.5, Kd_l = 0.15;

// Manual PID variables
double right_integral = 0.0, left_integral = 0.0;
double right_prev_error = 0.0, left_prev_error = 0.0;
double right_prev_meas = 0.0, left_prev_meas = 0.0;

// ================== CASTER SWIVEL PREVENTION ==================
bool needsPivot(float current_linear, float prev_linear) {
  // Check if we're changing from forward to backward or vice versa
  return (prev_linear > 0.1 && current_linear < -0.1) ||
         (prev_linear < -0.1 && current_linear > 0.1);
}

void startPivot(float linear_cmd, float angular_cmd) {
  pivot_mode = true;
  pivot_start_time = millis();
  stored_linear_cmd = linear_cmd;
  stored_angular_cmd = angular_cmd;

  // Determine pivot direction based on desired angular velocity
  // If no angular command, pivot in the direction that requires less rotation
  float pivot_angular = (angular_cmd != 0) ? (angular_cmd > 0 ? PIVOT_ANGULAR_VEL : -PIVOT_ANGULAR_VEL) : PIVOT_ANGULAR_VEL;

  // Execute pivot turn (no linear motion)
  cmdVelToWheels(0.0, pivot_angular);
}

bool updatePivot() {
  if (!pivot_mode) return false;

  if (millis() - pivot_start_time >= PIVOT_DURATION) {
    // Pivot complete, execute stored command
    pivot_mode = false;
    cmdVelToWheels(stored_linear_cmd, stored_angular_cmd);
    prev_linear_velocity = stored_linear_cmd;
    return false;
  }
  return true; // Still pivoting
}

// ================== PPM INTERRUPT ==================
void ppmISR() {
  unsigned long now = micros();
  unsigned long diff = now - lastMicros;
  lastMicros = now;

  if (diff > 3200) {
    ppmIndex = 0;
  } else {
    if (ppmIndex < CHANNELS) {
      ppmValues[ppmIndex] = diff;
      ppmIndex++;
    }
  }
}

// ================== PPM PROCESSING ==================
void processPPMInput() {
  // Check if PPM signal is active
  unsigned long current_time = millis();

  // Map PPM values to channels
  ch1 = ppmValues[0]; // Channel 1 - Angular (Rotation)
  ch2 = ppmValues[1]; // Channel 2 - Throttle (Linear)

  // Check for valid PPM signal (typical range 1000-2000)
  if (ch1 > 900 && ch1 < 2100 && ch2 > 900 && ch2 < 2100) {
    ppm_active = true;
    last_ppm_time = current_time;
    current_mode = PPM_MODE;

    // Apply deadband (center stick positions)
    int ch1_filtered = ch1;
    int ch2_filtered = ch2;

    if (ch1 > 1460 && ch1 < 1540) ch1_filtered = 1500;
    if (ch2 > 1460 && ch2 < 1540) ch2_filtered = 1500;

    // Constrain inputs to valid range for safety
    ch1_filtered = constrain(ch1_filtered, 1000, 2000);
    ch2_filtered = constrain(ch2_filtered, 1000, 2000);

    // Map Channel 2 to Linear Velocity
    float linear_velocity = ((float)(ch2_filtered - 1500) / 500.0) * PPM_MAX_LINEAR_VEL;
    linear_velocity = constrain(linear_velocity, -PPM_MAX_LINEAR_VEL, PPM_MAX_LINEAR_VEL);

    // Map Channel 1 to Angular Velocity
    float angular_velocity = ((float)(1500 - ch1_filtered) / 500.0) * PPM_MAX_ANGULAR_VEL;
    angular_velocity = constrain(angular_velocity, -PPM_MAX_ANGULAR_VEL, PPM_MAX_ANGULAR_VEL);

    // Check if pivot is needed for direction change
    if (needsPivot(linear_velocity, prev_linear_velocity) && !pivot_mode) {
      startPivot(linear_velocity, angular_velocity);
      return;
    }

    // If we're in pivot mode, continue pivoting
    if (updatePivot()) {
      return;
    }

    // Convert to wheel velocities using differential drive kinematics
    float left_linear_component = linear_velocity / WHEEL_RADIUS;
    float right_linear_component = linear_velocity / WHEEL_RADIUS;
    float angular_component = (angular_velocity * WHEELBASE / 2.0) / WHEEL_RADIUS;

    float left_wheel_vel = left_linear_component - angular_component;
    float right_wheel_vel = right_linear_component + angular_component;

    // Apply limits
    left_wheel_vel = constrain(left_wheel_vel, -MAX_ANGULAR_VEL, MAX_ANGULAR_VEL);
    right_wheel_vel = constrain(right_wheel_vel, -MAX_ANGULAR_VEL, MAX_ANGULAR_VEL);

    // Check for command changes to reset integrals
    if((right_wheel_cmd_vel * right_wheel_vel < 0) || (abs(right_wheel_vel - right_wheel_cmd_vel) > 3.0)) {
      right_integral = 0.0;
      right_wheel_target = 0.0;
    }

    if((left_wheel_cmd_vel * left_wheel_vel < 0) || (abs(left_wheel_vel - left_wheel_cmd_vel) > 3.0)) {
      left_integral = 0.0;
      left_wheel_target = 0.0;
    }

    // Update command velocities
    right_wheel_cmd_vel = right_wheel_vel;
    left_wheel_cmd_vel = left_wheel_vel;
    prev_linear_velocity = linear_velocity;

  } else {
    // Check for PPM timeout
    if (ppm_active && (current_time - last_ppm_time > PPM_TIMEOUT)) {
      ppm_active = false;
      pivot_mode = false; // Reset pivot mode on signal loss
      // Stop the wheelchair when PPM signal is lost
      right_wheel_cmd_vel = 0.0;
      left_wheel_cmd_vel = 0.0;
      prev_linear_velocity = 0.0;
      Serial.println("PPM signal lost - stopping");
    }
  }
}

// Encoder functions (unchanged)
void updateEncoderL(){
  int MSB = digitalRead(encoderPin1L);
  int LSB = digitalRead(encoderPin2L);
  int encoded = (MSB << 1) | LSB;
  int sum = (lastEncodedL << 2) | encoded;

  if(sum == 0b1101 || sum == 0b0100 || sum == 0b0010 || sum == 0b1011) encoderValueL++;
  if(sum == 0b1110 || sum == 0b0111 || sum == 0b0001 || sum == 0b1000) encoderValueL--;

  lastEncodedL = encoded;
}

void updateEncoderR(){
  int MSB = digitalRead(encoderPin1R);
  int LSB = digitalRead(encoderPin2R);
  int encoded = (MSB << 1) | LSB;
  int sum = (lastEncodedR << 2) | encoded;

  if(sum == 0b1101 || sum == 0b0100 || sum == 0b0010 || sum == 0b1011) encoderValueR++;
  if(sum == 0b1110 || sum == 0b0111 || sum == 0b0001 || sum == 0b1000) encoderValueR--;

  lastEncodedR = encoded;
}

// Forward kinematics: Convert cmd_vel to wheel velocities
void cmdVelToWheels(double linear, double angular) {
  // Check if pivot is needed for direction change
  if (needsPivot(linear, prev_linear_velocity) && !pivot_mode) {
    startPivot(linear, angular);
    return;
  }

  // Differential drive kinematics
  double right_vel = (linear + angular * WHEELBASE/2.0) / WHEEL_RADIUS;
  double left_vel = (linear - angular * WHEELBASE/2.0) / WHEEL_RADIUS;

  // Apply limits
  right_wheel_cmd_vel = constrain(right_vel, -MAX_ANGULAR_VEL, MAX_ANGULAR_VEL);
  left_wheel_cmd_vel = constrain(left_vel, -MAX_ANGULAR_VEL, MAX_ANGULAR_VEL);

  // Reset integrals on new command
  right_integral = 0.0;
  left_integral = 0.0;
  right_wheel_target = 0.0;
  left_wheel_target = 0.0;

  prev_linear_velocity = linear;
}

// Manual PID with anti-windup
double computePID(double setpoint, double measurement, double &integral,
                 double &prev_error, double &prev_meas, double Kp, double Ki, double Kd, double dt) {

  double error = setpoint - measurement;
  integral += error * dt;
  integral = constrain(integral, -100.0/Ki, 100.0/Ki);

  double derivative = -(measurement - prev_meas) / dt;
  double output = Kp * error + Ki * integral + Kd * derivative;

  double clamped_output = constrain(output, -100.0, 100.0);
  if (output != clamped_output) {
    integral = (clamped_output - Kp * error - Kd * derivative) / Ki;
  }

  prev_error = error;
  prev_meas = measurement;
  return clamped_output;
}

// Smooth acceleration limiting
double rampVelocity(double current, double target, double max_accel, double dt) {
  double diff = target - current;
  double max_change = max_accel * dt;

  if (abs(diff) <= max_change) {
    return target;
  } else {
    return current + (diff > 0 ? max_change : -max_change);
  }
}

void setup() {
  Serial.begin(115200);

  // PPM input setup
  pinMode(PPM_PIN, INPUT);
  attachInterrupt(digitalPinToInterrupt(PPM_PIN), ppmISR, RISING);

  // Relay initialization
  pinMode(relayPin, OUTPUT);
  digitalWrite(relayPin, HIGH);
  delay(5000);
  digitalWrite(relayPin, LOW);

  // Encoder setup
  pinMode(encoderPin1L, INPUT_PULLUP);
  pinMode(encoderPin2L, INPUT_PULLUP);
  pinMode(encoderPin1R, INPUT_PULLUP);
  pinMode(encoderPin2R, INPUT_PULLUP);

  attachInterrupt(digitalPinToInterrupt(encoderPin1L), updateEncoderL, CHANGE);
  attachInterrupt(digitalPinToInterrupt(encoderPin2L), updateEncoderL, CHANGE);
  attachInterrupt(digitalPinToInterrupt(encoderPin1R), updateEncoderR, CHANGE);
  attachInterrupt(digitalPinToInterrupt(encoderPin2R), updateEncoderR, CHANGE);

  filtered_right_vel = 0.0;
  filtered_left_vel = 0.0;
  last_millis = millis();

  Serial.println("Wheelchair Controller with PPM Support & Caster Swivel Prevention Ready");
  Serial.println("Commands:");
  Serial.println("Individual: rp5.0,lp3.0,");
  Serial.println("cmd_vel: x:0.5,t:0.1,");
  Serial.println("PPM: CH1=Angular, CH2=Throttle");
  Serial.println("PPM Priority: Will override serial when active");
  Serial.println("Caster Swivel Prevention: 300ms pivot on direction change");
}

void loop() {
  // Process PPM input first (highest priority)
  processPPMInput();

  // Handle pivot mode updates for serial commands
  if (!ppm_active && pivot_mode) {
    updatePivot();
  }

  // Only process serial commands if PPM is not active
  if (!ppm_active && Serial.available()) {
    char chr = Serial.read();

    // Detect command type
    if(chr == 'r' || chr == 'l') {
      current_mode = WHEEL_MODE;
    } else if(chr == 'x' || chr == 't') {
      current_mode = CMDVEL_MODE;
    }

    // WHEEL MODE PARSING
    if(current_mode == WHEEL_MODE) {
      if(chr == 'r') {
        is_right_wheel_cmd = true;
        is_left_wheel_cmd = false;
        value_idx = 0;
      }
      else if(chr == 'l') {
        is_right_wheel_cmd = false;
        is_left_wheel_cmd = true;
        value_idx = 0;
      }
      else if(chr == 'p') {
        if(is_right_wheel_cmd) {
          is_right_wheel_forward = true;
          right_wheel_sign = "p";
        }
        else if(is_left_wheel_cmd) {
          is_left_wheel_forward = true;
          left_wheel_sign = "p";
        }
      }
      else if(chr == 'n') {
        if(is_right_wheel_cmd) {
          is_right_wheel_forward = false;
          right_wheel_sign = "n";
        }
        else if(is_left_wheel_cmd) {
          is_left_wheel_forward = false;
          left_wheel_sign = "n";
        }
      }
      else if(chr == ',') {
        if(is_right_wheel_cmd) {
          double new_cmd = atof(value);
          if(!is_right_wheel_forward) new_cmd = -new_cmd;

          if((right_wheel_cmd_vel * new_cmd < 0) || (abs(new_cmd - right_wheel_cmd_vel) > 3.0)) {
            right_integral = 0.0;
            right_wheel_target = 0.0;
          }

          right_wheel_cmd_vel = constrain(new_cmd, -MAX_ANGULAR_VEL, MAX_ANGULAR_VEL);
        }
        else if(is_left_wheel_cmd) {
          double new_cmd = atof(value);
          if(!is_left_wheel_forward) new_cmd = -new_cmd;

          if((left_wheel_cmd_vel * new_cmd < 0) || (abs(new_cmd - left_wheel_cmd_vel) > 3.0)) {
            left_integral = 0.0;
            left_wheel_target = 0.0;
          }

          left_wheel_cmd_vel = constrain(new_cmd, -MAX_ANGULAR_VEL, MAX_ANGULAR_VEL);
        }

        // Reset value buffer
        value_idx = 0;
        value[0] = '0'; value[1] = '0'; value[2] = '.';
        value[3] = '0'; value[4] = '0'; value[5] = '\0';
      }
      else if(value_idx < 5) {
        value[value_idx] = chr;
        value_idx++;
      }
    }

    // CMD_VEL MODE PARSING
    else if(current_mode == CMDVEL_MODE) {
      if(chr == 'x') {
        is_linear_cmd = true;
        is_angular_cmd = false;
        value_idx = 0;
      }
      else if(chr == 't') {
        is_linear_cmd = false;
        is_angular_cmd = true;
        value_idx = 0;
      }
      else if(chr == ':') {
        // Start reading value after ':'
        value_idx = 0;
      }
      else if(chr == ',') {
        if(is_linear_cmd) {
          linear_x = constrain(atof(value), -MAX_LINEAR_VEL, MAX_LINEAR_VEL);
        }
        else if(is_angular_cmd) {
          angular_z = constrain(atof(value), -10.0, 10.0); // Max 10 rad/s angular

          // Apply forward kinematics when we get angular command
          cmdVelToWheels(linear_x, angular_z);
        }

        // Reset value buffer
        value_idx = 0;
        value[0] = '0'; value[1] = '0'; value[2] = '.';
        value[3] = '0'; value[4] = '0'; value[5] = '\0';
      }
      else if((chr >= '0' && chr <= '9') || chr == '.' || chr == '-') {
        if(value_idx < 5) {
          value[value_idx] = chr;
          value_idx++;
        }
      }
    }
  }

  // CONTROL LOOP
  unsigned long current_millis = millis();
  if(current_millis - last_millis >= interval) {
    float deltaT = (current_millis - last_millis) / 1000.0;

    // Calculate raw velocities
    long deltaCountsL = encoderValueL - lastEncoderL;
    long deltaCountsR = encoderValueR - lastEncoderR;

    double raw_right_vel = (deltaCountsR / (float)CPR) * (2.0 * PI / deltaT);
    double raw_left_vel = -(deltaCountsL / (float)CPR) * (2.0 * PI / deltaT);

    // Apply filtering
    filtered_right_vel = FILTER_ALPHA * raw_right_vel + (1.0 - FILTER_ALPHA) * filtered_right_vel;
    filtered_left_vel = FILTER_ALPHA * raw_left_vel + (1.0 - FILTER_ALPHA) * filtered_left_vel;

    right_wheel_meas_vel = filtered_right_vel;
    left_wheel_meas_vel = filtered_left_vel;

    // Apply smooth acceleration limiting
    right_wheel_target = rampVelocity(right_wheel_target, right_wheel_cmd_vel, MAX_ACCEL, deltaT);
    left_wheel_target = rampVelocity(left_wheel_target, left_wheel_cmd_vel, MAX_ACCEL, deltaT);

    // Manual PID with anti-windup
    right_wheel_cmd = computePID(right_wheel_target, right_wheel_meas_vel, right_integral,
                                right_prev_error, right_prev_meas, Kp_r, Ki_r, Kd_r, deltaT);

    left_wheel_cmd = computePID(left_wheel_target, left_wheel_meas_vel, left_integral,
                               left_prev_error, left_prev_meas, Kp_l, Ki_l, Kd_l, deltaT);

    // Handle zero commands
    if(abs(right_wheel_target) < 0.1) {
      right_wheel_cmd = 0.0;
      right_integral = 0.0;
    }
    if(abs(left_wheel_target) < 0.1) {
      left_wheel_cmd = 0.0;
      left_integral = 0.0;
    }

    // Update signs for feedback
    right_wheel_sign = (right_wheel_meas_vel >= 0) ? "p" : "n";
    left_wheel_sign = (left_wheel_meas_vel >= 0) ? "p" : "n";

    // Apply motor commands
    smartDriveDuo.control((int)right_wheel_cmd, -(int)left_wheel_cmd);

    // Enhanced feedback - show current mode and velocities
    String mode_indicator = "";
    if (ppm_active) {
      mode_indicator = "[PPM] ";
    } else if (current_mode == CMDVEL_MODE) {
      mode_indicator = "[CMD_VEL] ";
    } else {
      mode_indicator = "[WHEEL] ";
    }

    if (pivot_mode) {
      mode_indicator += "[PIVOT] ";
    }

    String encoder_read = mode_indicator + "r" + right_wheel_sign + String(abs(right_wheel_meas_vel), 2) +
                         ",l" + left_wheel_sign + String(abs(left_wheel_meas_vel), 2) + ",";
    Serial.println(encoder_read);

    // Update for next iteration
    lastEncoderL = encoderValueL;
    lastEncoderR = encoderValueR;
    last_millis = current_millis;
  }
}