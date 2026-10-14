# Phase 2B — Domain alignment (US 311 → UrbanEye)

Generated 2026-10-03T19:46:35.028199+00:00 by `scripts/phase2b/domain_alignment.py`.

## UrbanEye categories

| UrbanEye | Status | Rows | Sources | Default SLA | Median resolution by source (h) | Resolved within SLA by source |
|---|---|---|---|---|---|---|
| POTHOLE | SUPPORTED | 157,907 | {'chicago311': 138255, 'nyc311': 16802, 'sf311': 2850} | P3 / 72 h | {'chicago311': 332.4, 'nyc311': 36.9, 'sf311': 115.3} | {'chicago311': 0.248, 'nyc311': 0.656, 'sf311': 0.379} |
| FLOOD | SUPPORTED | 19,732 | {'chicago311': 17317, 'nyc311': 1610, 'sf311': 805} | P2 / 24 h | {'chicago311': 53.2, 'nyc311': 51.7, 'sf311': 69.0} | {'chicago311': 0.358, 'nyc311': 0.349, 'sf311': 0.38} |
| FIRE_HAZARD | UNSUPPORTED (no training data) | 0 | {} | P1 / 4 h | {} | {} |
| GARBAGE | SUPPORTED | 114,644 | {'sf311': 56436, 'chicago311': 55305, 'nyc311': 2903} | P4 / 168 h | {'chicago311': 93.1, 'nyc311': 91.8, 'sf311': 2.6} | {'chicago311': 0.66, 'nyc311': 0.739, 'sf311': 0.963} |
| STREETLIGHT | SUPPORTED | 162,330 | {'chicago311': 140211, 'nyc311': 19696, 'sf311': 2423} | P3 / 72 h | {'chicago311': 554.3, 'nyc311': 41.8, 'sf311': 68.7} | {'chicago311': 0.065, 'nyc311': 0.601, 'sf311': 0.521} |
| OTHER | UNSUPPORTED (no training data) | 0 | {} | P3 / 72 h | {} | {} |

## Mapping rules (raw publisher category → UrbanEye)

| Source | Raw category | Subcategory | → UrbanEye | Confidence | Rows | Notes |
|---|---|---|---|---|---|---|
| chicago311 | Pothole in Street Complaint | * | POTHOLE | EXACT | 116,465 |  |
| chicago311 | Alley Pothole Complaint | * | POTHOLE | EXACT | 21,790 | alley rather than street |
| nyc311 | Street Condition | Pothole | POTHOLE | EXACT | 16,085 |  |
| nyc311 | Highway Condition | Pothole - Highway | POTHOLE | EXACT | 717 |  |
| sf311 | Street Defects | Pavement_Defect | POTHOLE | PARTIAL | 2,850 | pavement defects include cracks and sunken pavement as well as potholes |
| chicago311 | Street Light Out Complaint | * | STREETLIGHT | EXACT | 140,211 |  |
| nyc311 | Street Light Condition | * | STREETLIGHT | EXACT | 19,696 | includes cycling/dayburning lamps |
| sf311 | Streetlights | * | STREETLIGHT | EXACT | 2,423 |  |
| chicago311 | Fly Dumping Complaint | * | GARBAGE | EXACT | 16,272 | illegal dumping on public way |
| chicago311 | Missed Garbage Pick-Up Complaint | * | GARBAGE | STRONG | 9,863 | uncollected garbage |
| chicago311 | Sanitation Code Violation | * | GARBAGE | PARTIAL | 22,401 | dumping/overflowing/improper storage plus other code violations |
| chicago311 | Street Cleaning Request | * | GARBAGE | PARTIAL | 6,769 | litter/debris on street; sweeping is a scheduled service |
| chicago311 | Garbage Cart Maintenance | * | UNSUPPORTED | — | 84,697 | cart repair/delivery — not a garbage complaint |
| sf311 | Street and Sidewalk Cleaning | City_garbage_can_overflowing | GARBAGE | EXACT | 4,970 |  |
| sf311 | Street and Sidewalk Cleaning | Bulky Items | GARBAGE | STRONG | 29,499 | dumped bulky items on street |
| sf311 | Street and Sidewalk Cleaning | General Cleaning | GARBAGE | STRONG | 21,950 | litter/debris on street or sidewalk |
| sf311 | Street and Sidewalk Cleaning | * | UNSUPPORTED | — | 15,582 | human/animal waste; hazardous or medical waste — no UrbanEye equivalent |
| sf311 | Litter Receptacles | Debris_box_maintenance_overflowing | GARBAGE | STRONG | 17 |  |
| sf311 | Litter Receptacles | * | UNSUPPORTED | — | 2,676 | bin repair/maintenance — not a garbage complaint |
| nyc311 | Dirty Conditions | E15 Illegal Postering | UNSUPPORTED | — | 1,184 | posters — not garbage |
| nyc311 | Dirty Conditions | E8 Canine Violation | UNSUPPORTED | — | 974 | dog waste — not garbage |
| nyc311 | Dirty Conditions | E4 18" Law | UNSUPPORTED | — | 158 | storefront sweeping ordinance |
| nyc311 | Dirty Conditions | * | GARBAGE | PARTIAL | 2,903 | improper disposal / dumping / loose rubbish / dirty sidewalk |
| chicago311 | Water On Street Complaint | * | FLOOD | STRONG | 17,317 | water pooling or flowing on the street (waterlogging); includes main-break water |
| nyc311 | Sewer | Street Flooding (SJ) | FLOOD | EXACT | 352 |  |
| nyc311 | Sewer | Catch Basin Clogged/Flooding (Use Comments) (SC) | FLOOD | STRONG | 1,151 | clogged drain with flooding |
| nyc311 | Sewer | Manhole Overflow (Use Comments) (SA1) | FLOOD | PARTIAL | 107 | sewer overflow onto street |
| sf311 | Sewer Issues | Flooding | FLOOD | EXACT | 805 |  |
| chicago311 | Water in Basement Complaint | * | UNSUPPORTED | — | 9,689 | private-property water — not a civic flood report |
| chicago311 | Open Fire Hydrant Complaint | * | UNSUPPORTED | — | 12,633 | hydrant running — a different service |
| chicago311 | Sewer Cleaning Inspection Request | * | UNSUPPORTED | — | 25,564 | drain inspection; flooding not established from the record |
| chicago311 | Commercial Fire Safety Inspection Request | * | UNSUPPORTED | — | 710 | inspection request — not a citizen fire-hazard report; 710 rows |
| sf311 | General Request - FIRE DEPARTMENT | * | UNSUPPORTED | — | 34 | 34 rows; content unknown |

## Canonical categories with no UrbanEye equivalent

Used only as auxiliary training rows; in UrbanEye these would be filed as OTHER, whose composition in Bengaluru is unknown.

- GRAFFITI_VISUAL_POLLUTION: 177,920
- FALLEN_TREE: 115,771
- GARBAGE_DUMPING: 89,689
- SIGNAGE_DAMAGE: 58,179
- TRAFFIC_SIGNAL_FAULT: 47,234
- DRAINAGE_SEWER: 47,185
- WATER_LEAKAGE_WATERLOGGING: 30,714
- FOOTPATH_DAMAGE: 20,845
- REVIEW_REQUIRED: 12,107
- ROAD_DAMAGE: 9,549
- PUBLIC_INFRASTRUCTURE_DAMAGE: 9,480

## Geography

- **chicago311**: lat [41.645, 42.023], lon [-87.931, -87.524], period ['2018-07-01', '2020-08-28'], zones ['ward'], rows inside Bengaluru bbox: 0
- **nyc311**: lat [40.5, 40.912], lon [-74.253, -73.701], period ['2010-01-01', '2010-03-09'], zones ['community_board'], rows inside Bengaluru bbox: 0
- **sf311**: lat [37.708, 37.831], lon [-122.514, -122.364], period ['2018-01-01', '2018-05-10'], zones ['analysis_neighborhood'], rows inside Bengaluru bbox: 0
- UrbanEye DB incidents by bbox: {'Chennai': 3, 'Bengaluru': 3}

## Source / department / zones / SLA

- Channel: no defensible mapping; report_channel excluded from deployable models
- Department: excluded from deployable models
- Zones: deployable models use no zone; hotspot v2 uses a 1 km grid computed from location
- SLA (UrbanEye): {'P1': 4, 'P2': 24, 'P3': 72, 'P4': 168}; values in DB: {'72': 2, '24': 1, '12': 1, '4': 2}
- Phase 2A SLA target: NYC agency Due Date vs closed_date (median deadline 240 h), open cases dropped
- Phase 2B SLA target: UrbanEye default deadline of the mapped category; open past deadline = breach
