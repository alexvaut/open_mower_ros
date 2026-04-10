// Blade Speed Adapter — dynamically adjusts mower travel speed based on blade RPM.
//
// Subscribes to blade motor RPM from /ll/mower_status and mowing state from
// mower_logic/current_state. Computes an adaptive speed_fast for the FTC local
// planner: slows down instantly when the blade bogs, recovers gradually.
//
// In dry-run mode (default), only logs computed values without touching the planner.
// In live mode (~dry_run:=false), pushes speed changes via dynamic_reconfigure.

#include <ros/ros.h>
#include <dynamic_reconfigure/client.h>
#include <ftc_local_planner/FTCPlannerConfig.h>
#include <mower_msgs/HighLevelStatus.h>
#include <mower_msgs/Status.h>
#include <std_msgs/String.h>

#include <algorithm>
#include <deque>
#include <mutex>
#include <sstream>

// ---------------------------------------------------------------------------
// Global state
// ---------------------------------------------------------------------------

static std::mutex g_mutex;

// Latest sensor data
static mower_msgs::Status g_status;
static bool g_has_status = false;

// Latest high-level state
static mower_msgs::HighLevelStatus g_high_level;
static bool g_has_high_level = false;

// FTC planner config
static dynamic_reconfigure::Client<ftc_local_planner::FTCPlannerConfig>* g_ftc_client = nullptr;
static ftc_local_planner::FTCPlannerConfig g_ftc_config;
static bool g_has_ftc_config = false;
static double g_original_speed_fast = -1.0;
static double g_original_speed_slow = -1.0;

// Algorithm state
static std::deque<double> g_rpm_buffer;
static double g_actual_speed = 0.0;
static bool g_was_mowing = false;

// ROS parameters
static bool p_dry_run;
static double p_rpm_nominal;
static double p_rpm_min_safe;
static double p_speed_max;
static double p_speed_min;
static double p_recovery_rate;
static double p_sample_interval;
static int p_rpm_window;

// Publisher
static ros::Publisher g_log_pub;

// ---------------------------------------------------------------------------
// Callbacks
// ---------------------------------------------------------------------------

static void statusCallback(const mower_msgs::Status::ConstPtr& msg) {
  std::lock_guard<std::mutex> lock(g_mutex);
  g_status = *msg;
  g_has_status = true;
}

static void highLevelCallback(const mower_msgs::HighLevelStatus::ConstPtr& msg) {
  std::lock_guard<std::mutex> lock(g_mutex);
  g_high_level = *msg;
  g_has_high_level = true;
}

static void ftcConfigCallback(const ftc_local_planner::FTCPlannerConfig& config) {
  std::lock_guard<std::mutex> lock(g_mutex);
  g_ftc_config = config;
  g_has_ftc_config = true;

  // Capture the original values on first callback (before we modify anything)
  if (g_original_speed_fast < 0.0) {
    g_original_speed_fast = config.speed_fast;
    g_original_speed_slow = config.speed_slow;
    ROS_INFO("blade_speed_adapter: captured original speed_fast=%.3f speed_slow=%.3f",
             g_original_speed_fast, g_original_speed_slow);
  }
}

// ---------------------------------------------------------------------------
// Restore original planner speeds
// ---------------------------------------------------------------------------

static void restoreOriginalSpeed() {
  if (p_dry_run || g_original_speed_fast < 0.0) return;

  ftc_local_planner::FTCPlannerConfig cfg;
  {
    std::lock_guard<std::mutex> lock(g_mutex);
    cfg = g_ftc_config;
  }
  cfg.speed_fast = g_original_speed_fast;
  cfg.speed_slow = g_original_speed_slow;
  g_ftc_client->setConfiguration(cfg);
  ROS_INFO("blade_speed_adapter: restored speed_fast=%.3f speed_slow=%.3f",
           g_original_speed_fast, g_original_speed_slow);
}

// ---------------------------------------------------------------------------
// Control loop
// ---------------------------------------------------------------------------

static void controlLoop(const ros::TimerEvent&) {
  // Snapshot current state under lock
  mower_msgs::Status status;
  mower_msgs::HighLevelStatus high_level;
  bool has_data;
  {
    std::lock_guard<std::mutex> lock(g_mutex);
    status = g_status;
    high_level = g_high_level;
    has_data = g_has_status && g_has_high_level && g_has_ftc_config;
  }

  if (!has_data) {
    ROS_INFO_THROTTLE(10, "blade_speed_adapter: waiting for data (status=%d hl=%d ftc=%d)",
                      g_has_status, g_has_high_level, g_has_ftc_config);
    return;
  }

  bool is_mowing = (high_level.state_name == "MOWING") && status.mow_enabled;

  // Handle mowing → not-mowing transition
  if (!is_mowing) {
    if (g_was_mowing) {
      ROS_INFO("blade_speed_adapter: mowing stopped, restoring original speeds");
      restoreOriginalSpeed();
      g_rpm_buffer.clear();
      g_actual_speed = p_speed_max;
      g_was_mowing = false;
    }
    return;
  }

  g_was_mowing = true;

  // Push RPM into moving average buffer
  double rpm = static_cast<double>(status.mower_motor_rpm);
  g_rpm_buffer.push_back(rpm);
  while (static_cast<int>(g_rpm_buffer.size()) > p_rpm_window) {
    g_rpm_buffer.pop_front();
  }

  // Compute smoothed RPM
  double rpm_sum = 0.0;
  for (double r : g_rpm_buffer) {
    rpm_sum += r;
  }
  double rpm_avg = rpm_sum / static_cast<double>(g_rpm_buffer.size());

  // Compute load ratio and target speed
  double load_ratio = 0.0;
  if (rpm_avg > 0.0) {
    load_ratio = (rpm_avg - p_rpm_min_safe) / (p_rpm_nominal - p_rpm_min_safe);
    load_ratio = std::max(0.0, std::min(1.0, load_ratio));
  }
  double target_speed = p_speed_min + load_ratio * (p_speed_max - p_speed_min);

  // Asymmetric ramp: instant slowdown, gradual recovery
  if (target_speed < g_actual_speed) {
    g_actual_speed = target_speed;
  } else {
    g_actual_speed += std::min(p_recovery_rate * p_sample_interval,
                               target_speed - g_actual_speed);
  }

  // Publish diagnostic log (always, regardless of dry_run)
  {
    std_msgs::String log_msg;
    std::ostringstream ss;
    ss << "rpm=" << rpm
       << ";rpm_avg=" << rpm_avg
       << ";current=" << status.mower_esc_current
       << ";load_ratio=" << load_ratio
       << ";target_speed=" << target_speed
       << ";actual_speed=" << g_actual_speed
       << ";mode=" << (p_dry_run ? "dry_run" : "live")
       << ";area=" << high_level.current_area
       << ";path=" << high_level.current_path;
    log_msg.data = ss.str();
    g_log_pub.publish(log_msg);
  }

  ROS_INFO_THROTTLE(5, "blade_speed_adapter [%s]: RPM=%.0f avg=%.0f load=%.2f "
                       "target=%.3f actual=%.3f m/s [area %d path %d]",
                    p_dry_run ? "dry_run" : "live",
                    rpm, rpm_avg, load_ratio, target_speed, g_actual_speed,
                    high_level.current_area, high_level.current_path);

  // Push speed to FTC planner in live mode
  if (!p_dry_run) {
    ftc_local_planner::FTCPlannerConfig cfg;
    {
      std::lock_guard<std::mutex> lock(g_mutex);
      cfg = g_ftc_config;
    }
    cfg.speed_fast = g_actual_speed;
    cfg.speed_slow = std::min(g_actual_speed, g_original_speed_slow);
    g_ftc_client->setConfiguration(cfg);
  }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
  ros::init(argc, argv, "blade_speed_adapter");
  ros::NodeHandle n;
  ros::NodeHandle pn("~");

  // Load parameters
  pn.param("dry_run", p_dry_run, true);
  pn.param("rpm_nominal", p_rpm_nominal, 3800.0);
  pn.param("rpm_min_safe", p_rpm_min_safe, 2300.0);
  pn.param("speed_max", p_speed_max, 0.4);
  pn.param("speed_min", p_speed_min, 0.05);
  pn.param("recovery_rate", p_recovery_rate, 0.02);
  pn.param("sample_interval", p_sample_interval, 0.5);
  pn.param("rpm_window", p_rpm_window, 5);

  g_actual_speed = p_speed_max;

  ROS_INFO("blade_speed_adapter: starting in %s mode", p_dry_run ? "DRY-RUN" : "LIVE");
  ROS_INFO("  rpm_nominal=%.0f  rpm_min_safe=%.0f", p_rpm_nominal, p_rpm_min_safe);
  ROS_INFO("  speed=[%.3f, %.3f] m/s  recovery_rate=%.3f m/s/s", p_speed_min, p_speed_max, p_recovery_rate);
  ROS_INFO("  sample_interval=%.2fs  rpm_window=%d", p_sample_interval, p_rpm_window);

  // Subscribers
  ros::Subscriber status_sub = n.subscribe("/ll/mower_status", 10, statusCallback);
  ros::Subscriber hl_sub = n.subscribe("mower_logic/current_state", 10, highLevelCallback);

  // Diagnostic publisher
  g_log_pub = pn.advertise<std_msgs::String>("log", 50);

  // FTC planner dynamic_reconfigure client
  g_ftc_client = new dynamic_reconfigure::Client<ftc_local_planner::FTCPlannerConfig>(
      "/move_base_flex/FTCPlanner", ftcConfigCallback);

  // Control loop timer
  ros::Timer timer = n.createTimer(ros::Duration(p_sample_interval), controlLoop);

  ros::spin();

  // Cleanup: restore original speed if we were actively controlling
  if (g_was_mowing) {
    restoreOriginalSpeed();
  }
  delete g_ftc_client;

  return 0;
}
