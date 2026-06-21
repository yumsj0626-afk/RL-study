# Failure Taxonomy (offline)

계획 단계에서 의도적으로 중단된(controlled) 케이스:
- **fail_goal_in_obstacle**: Goal (4.0, 4.0) is inside an inflated forbidden region
- **fail_no_path**: No collision-free path from start to goal
- **fail_start_equals_goal**: Start equals goal
- **fail_start_in_obstacle**: Start (4.0, 4.0) is inside an inflated forbidden region
- **underspecified**: Goal is underspecified (goal=null)