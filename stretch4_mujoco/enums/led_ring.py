from enum import Enum

#: Names of the MJCF assets `mjcf_generator` emits for the head LED rings.
#: Both rings share one material, so recolouring it drives both.
LED_RING_MESH = "led_ring"
LED_RING_MATERIAL = "led_ring"


class LedRingAnimation(str, Enum):
    """How the LED rings around Stretch 4's head cameras are driven.

    The rings are visual only -- nothing in the physics depends on them.
    """

    #: Hold the colour steady.
    solid = "solid"
    #: Fade smoothly between off and the colour, once per period.
    pulse = "pulse"
    #: Switch between off and the colour, half the period each.
    blink = "blink"
    #: Dark rings.
    off = "off"
