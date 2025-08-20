# Duration Measurement, Logging, and Hazard Detection Module

"""1. Duration Measurement (Time Tracking)
We will store the time each person (ID) starts being tracked and the last time they were seen.
This allows us to measure how many seconds the person has been in the frame and how much time has passed.
Using the track_id values from the HumanTracker class, this information can be stored in a dictionary.
"""

"""2. CSV/JSON Logging
The collected information (ID, time, position, status, etc.) can be saved to a CSV or JSON file.
For example, logs can be recorded for each frame or at specific intervals."""

"""3. Hazard Detection
If a person remains motionless for a long time (could indicate a fall or unconsciousness),
When they enter a restricted area,
When they move too quickly (could indicate falling or running),
We can use track_history here to measure movement speed and position changes."""
