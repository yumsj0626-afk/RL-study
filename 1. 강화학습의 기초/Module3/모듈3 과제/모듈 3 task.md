과제 : RL로 풀 수 있는 문제 3개를 직접 만들기. 단 서로 다른 도메인으로.

MDP Task: Hospital Patient Triage System

State
The state at each time step consists of:

Patient's vital signs (heart rate, blood pressure, oxygen saturation, body temperature)
Patient's reported pain level (0–10 scale)
Number of patients currently waiting in the emergency room
Number of available doctors and nurses on shift
Time elapsed since the patient arrived

Action
The agent can choose one of the following:

Assign the patient to immediate emergency care
Assign the patient to urgent care (seen within 30 minutes)
Assign the patient to standard care (seen within 2 hours)
Request additional diagnostic tests before deciding

Reward

+100 if a critical patient is correctly identified and treated immediately
+10 if a non-critical patient is assigned to the appropriate care level
-50 if a critical patient is under-triaged and their condition worsens
-5 for each unnecessary emergency assignment (wasting resources on non-critical patients)
-1 at every time step a patient waits without being assigned (encourages efficiency)
---
MDP Task 2: Personalized Music Playlist Curator

State
The state at each time step consists of:

User's current heart rate (from wearable device)
Time of day and day of the week
User's recent skip rate (how often they skipped songs in the last 10 minutes)
Current song's genre, tempo (BPM), and energy level
User's location context (home, gym, commuting)

Action
The agent can choose one of the following:

Play the next song in the current genre
Switch to a higher energy / faster tempo song
Switch to a lower energy / slower tempo song
Introduce a new genre the user hasn't heard recently
Repeat the current song

Reward

+20 if the user listens to a song all the way through without skipping
+10 if the user manually likes or saves the song
-15 if the user skips the song within the first 10 seconds
-5 if the user skips the song before it finishes
+5 if the song's energy level matches the user's heart rate context (e.g., high BPM at the gym)
---
MDP Task 3: Agricultural Irrigation Control System

State
The state at each time step consists of:

Current soil moisture level across different field zones (%)
Weather forecast for the next 24 hours (precipitation probability, temperature)
Crop growth stage (seedling / vegetative / flowering / harvest)
Current water reservoir level
Time since last irrigation

Action
The agent can choose one of the following:

Irrigate a specific field zone at high intensity
Irrigate a specific field zone at low intensity
Skip irrigation for this time step
Redistribute water from a low-priority zone to a high-priority zone

Reward

+50 if crop yield at harvest meets or exceeds the target
+10 if soil moisture is maintained within the optimal range for the current growth stage
-20 if soil moisture drops below the critical threshold (crop stress)
-10 if water is wasted by irrigating right before a rainfall event
-30 if the water reservoir runs empty