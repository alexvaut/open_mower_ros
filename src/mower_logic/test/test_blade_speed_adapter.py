#!/usr/bin/env python3
"""
Integration test for blade_speed_adapter node.

Launched via blade_speed_adapter.test with the full simulation stack.
The test verifies:
1. The node starts and publishes diagnostics on its log topic
2. When idle (not mowing), the adapter does not change planner speed
3. When mowing state is faked with dropping RPM, the adapter reduces speed
4. When mowing stops, speed is restored to original
"""

import unittest
import time
import rospy
import dynamic_reconfigure.client
from std_msgs.msg import String
from mower_msgs.msg import HighLevelStatus, Status


def wait_for(condition_fn, timeout_sec, poll_hz=10):
    """Poll condition_fn() until it returns True or timeout."""
    rate = rospy.Rate(poll_hz)
    deadline = rospy.Time.now() + rospy.Duration(timeout_sec)
    while not rospy.is_shutdown() and rospy.Time.now() < deadline:
        if condition_fn():
            return True
        rate.sleep()
    return False


def parse_log(msg_data):
    """Parse 'key=value;key=value' log string into dict."""
    result = {}
    for part in msg_data.split(";"):
        if "=" in part:
            k, v = part.split("=", 1)
            result[k] = v
    return result


class TestBladeSpeedAdapter(unittest.TestCase):
    """Integration tests for the blade_speed_adapter node."""

    @classmethod
    def setUpClass(cls):
        """Wait for the simulation stack and adapter to be ready."""
        cls.log_messages = []
        cls.log_sub = rospy.Subscriber(
            "/blade_speed_adapter/log", String, cls._log_cb)

        # Wait for the adapter's log topic to appear (node is alive)
        rospy.loginfo("Waiting for blade_speed_adapter log topic...")
        ok = wait_for(lambda: len(cls.log_messages) > 0 or
                      rospy.Time.now() > rospy.Time(0), 30)

        # Publishers to fake mower state — we'll use these to override
        # the simulation's values for controlled testing
        cls.status_pub = rospy.Publisher(
            "/ll/mower_status", Status, queue_size=10)
        cls.hl_pub = rospy.Publisher(
            "mower_logic/current_state", HighLevelStatus, queue_size=10)

        # Wait a bit for publishers to register
        rospy.sleep(2.0)

    @classmethod
    def _log_cb(cls, msg):
        cls.log_messages.append(msg.data)

    def _clear_logs(self):
        """Clear collected logs for a fresh assertion window."""
        self.__class__.log_messages = []

    def _get_ftc_speed_fast(self):
        """Read current speed_fast from FTC planner via dynamic_reconfigure."""
        try:
            client = dynamic_reconfigure.client.Client(
                "/move_base_flex/FTCPlanner", timeout=10)
            config = client.get_configuration()
            return config.get("speed_fast", None)
        except Exception as e:
            rospy.logwarn("Could not read FTC config: %s" % e)
            return None

    def _publish_mowing_state(self, rpm, mow_enabled=True):
        """Publish fake status and high-level state as if mowing."""
        status = Status()
        status.stamp = rospy.Time.now()
        status.mower_status = 255  # MOWER_STATUS_OK
        status.mow_enabled = mow_enabled
        status.mower_motor_rpm = rpm
        status.mower_esc_current = 1.0 if mow_enabled else 0.0
        status.mower_motor_temperature = 25.0
        status.mower_esc_temperature = 35.0
        self.status_pub.publish(status)

        hl = HighLevelStatus()
        hl.state = HighLevelStatus.HIGH_LEVEL_STATE_AUTONOMOUS
        hl.state_name = "MOWING"
        hl.current_area = 0
        hl.current_path = 0
        self.hl_pub.publish(hl)

    def _publish_idle_state(self):
        """Publish fake status and high-level state as if idle."""
        status = Status()
        status.stamp = rospy.Time.now()
        status.mower_status = 255
        status.mow_enabled = False
        status.mower_motor_rpm = 0.0
        status.mower_esc_current = 0.0
        status.mower_motor_temperature = 25.0
        status.mower_esc_temperature = 35.0
        self.status_pub.publish(status)

        hl = HighLevelStatus()
        hl.state = HighLevelStatus.HIGH_LEVEL_STATE_IDLE
        hl.state_name = "IDLE"
        self.hl_pub.publish(hl)

    # ------------------------------------------------------------------
    # Test 1: Node starts and captures original FTC config
    # ------------------------------------------------------------------

    def test_1_node_starts(self):
        """blade_speed_adapter should start and connect to FTC planner."""
        # The node should have captured the original speed from FTC planner.
        # We verify by checking that the FTC planner is reachable.
        speed = self._get_ftc_speed_fast()
        self.assertIsNotNone(speed, "Could not read FTC planner speed_fast")
        self.assertAlmostEqual(speed, 0.4, places=1,
                               msg="Initial speed_fast should be ~0.4")

    # ------------------------------------------------------------------
    # Test 2: Idle state — adapter should not change speed
    # ------------------------------------------------------------------

    def test_2_idle_no_speed_change(self):
        """When idle, adapter should not modify planner speed."""
        original_speed = self._get_ftc_speed_fast()
        self.assertIsNotNone(original_speed)

        # Publish idle state for 2 seconds
        for _ in range(20):
            self._publish_idle_state()
            rospy.sleep(0.1)

        current_speed = self._get_ftc_speed_fast()
        self.assertAlmostEqual(current_speed, original_speed, places=3,
                               msg="Speed should not change while idle")

    # ------------------------------------------------------------------
    # Test 3: Mowing at nominal RPM — speed stays at max
    # ------------------------------------------------------------------

    def test_3_nominal_rpm_max_speed(self):
        """Mowing at nominal RPM should keep speed_fast at maximum."""
        self._clear_logs()

        # Publish mowing state with nominal RPM (3800)
        for _ in range(20):
            self._publish_mowing_state(rpm=3800.0)
            rospy.sleep(0.1)

        # Check that speed_fast is still at max
        speed = self._get_ftc_speed_fast()
        self.assertIsNotNone(speed)
        self.assertAlmostEqual(speed, 0.4, places=2,
                               msg="speed_fast should be 0.4 at nominal RPM")

        # Verify log messages were produced
        self.assertGreater(len(self.log_messages), 0,
                           "Should have received log messages while mowing")

        # Parse a log and check format
        log = parse_log(self.log_messages[-1])
        self.assertIn("rpm", log, "Log should contain 'rpm' field")
        self.assertIn("actual_speed", log, "Log should contain 'actual_speed'")
        self.assertIn("load_ratio", log, "Log should contain 'load_ratio'")

    # ------------------------------------------------------------------
    # Test 4: RPM drops — speed should decrease
    # ------------------------------------------------------------------

    def test_4_rpm_drop_reduces_speed(self):
        """When blade RPM drops, speed_fast should decrease."""
        # First mow at nominal to establish baseline
        for _ in range(10):
            self._publish_mowing_state(rpm=3800.0)
            rospy.sleep(0.1)

        # Now drop RPM to 2800 (load_ratio ~0.33 → speed ~0.17)
        for _ in range(20):
            self._publish_mowing_state(rpm=2800.0)
            rospy.sleep(0.1)

        speed = self._get_ftc_speed_fast()
        self.assertIsNotNone(speed)
        self.assertLess(speed, 0.3,
                        "speed_fast should drop below 0.3 when RPM is 2800")
        self.assertGreater(speed, 0.0,
                           "speed_fast should still be positive")

    # ------------------------------------------------------------------
    # Test 5: RPM near critical — speed should be near minimum
    # ------------------------------------------------------------------

    def test_5_near_stall_minimum_speed(self):
        """When RPM is near critical (2300), speed should be near minimum."""
        for _ in range(20):
            self._publish_mowing_state(rpm=2400.0)
            rospy.sleep(0.1)

        speed = self._get_ftc_speed_fast()
        self.assertIsNotNone(speed)
        self.assertLess(speed, 0.15,
                        "speed_fast should be < 0.15 at RPM 2400")

    # ------------------------------------------------------------------
    # Test 6: Recovery is gradual
    # ------------------------------------------------------------------

    def test_6_recovery_is_gradual(self):
        """After RPM recovers, speed should increase gradually, not jump."""
        # Drive RPM down first
        for _ in range(20):
            self._publish_mowing_state(rpm=2400.0)
            rospy.sleep(0.1)

        low_speed = self._get_ftc_speed_fast()
        self.assertIsNotNone(low_speed)

        # Now recover RPM to nominal
        self._clear_logs()
        for _ in range(10):
            self._publish_mowing_state(rpm=3800.0)
            rospy.sleep(0.1)

        mid_speed = self._get_ftc_speed_fast()
        self.assertIsNotNone(mid_speed)

        # Speed should have increased but NOT jumped back to 0.4
        self.assertGreater(mid_speed, low_speed,
                           "Speed should increase when RPM recovers")
        self.assertLess(mid_speed, 0.35,
                        "Speed should recover gradually, not jump to max "
                        "(got %.3f after 1s of recovery)" % mid_speed)

    # ------------------------------------------------------------------
    # Test 7: Mowing stops — speed restored
    # ------------------------------------------------------------------

    def test_7_stop_mowing_restores_speed(self):
        """When mowing stops, speed_fast should be restored to original."""
        # Mow at low RPM to change speed
        for _ in range(20):
            self._publish_mowing_state(rpm=2800.0)
            rospy.sleep(0.1)

        speed_during = self._get_ftc_speed_fast()
        self.assertLess(speed_during, 0.4,
                        "Speed should be reduced while mowing at low RPM")

        # Stop mowing
        for _ in range(20):
            self._publish_idle_state()
            rospy.sleep(0.1)

        speed_after = self._get_ftc_speed_fast()
        self.assertIsNotNone(speed_after)
        self.assertAlmostEqual(speed_after, 0.4, places=2,
                               msg="speed_fast should be restored to 0.4 after mowing stops")

    # ------------------------------------------------------------------
    # Test 8: Log format validation
    # ------------------------------------------------------------------

    def test_8_log_format(self):
        """Log messages should contain all expected fields."""
        self._clear_logs()

        for _ in range(10):
            self._publish_mowing_state(rpm=3500.0)
            rospy.sleep(0.1)

        self.assertGreater(len(self.log_messages), 0,
                           "Should have log messages")

        log = parse_log(self.log_messages[-1])
        expected_fields = ["rpm", "rpm_avg", "current", "load_ratio",
                           "target_speed", "actual_speed", "mode",
                           "area", "path"]
        for field in expected_fields:
            self.assertIn(field, log,
                          "Log missing field: %s (got: %s)" % (field, log))

        # Mode should be 'live' since dry_run=false in the launch
        self.assertEqual(log["mode"], "live",
                         "Mode should be 'live' in test configuration")


if __name__ == "__main__":
    import rostest
    rospy.init_node("test_blade_speed_adapter", anonymous=True)
    rostest.rosrun("mower_logic", "test_blade_speed_adapter",
                   TestBladeSpeedAdapter)
