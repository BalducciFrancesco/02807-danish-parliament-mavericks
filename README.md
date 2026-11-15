# Danish Parliament Mavericks (02807 Computational Tools for Data Science)

**Which MPs vote more like another party than their own?**

Course project for DTU 02807 *Computational Tools for Data Science*. We use roll-call votes from the
Danish Parliament (Folketinget) in the 2022–25 legislature to group MPs by how they actually vote, and
to find "mavericks": MPs whose votes put them closer to another party than to their own.

## Approach

1. **Data collection & preprocessing**: download every vote and each MP's ballot from the
   [Folketinget Open Data API](https://oda.ft.dk/api/), and attach the party each MP belonged to on the day of each vote.
2. **Network + Louvain**: build an MP–MP similarity network from voting agreement and detect communities.
3. **Alternative clustering**: DBSCAN / Spectral clustering, compared against Louvain.
4. **Maverick detection**: compare each MP with their own party and with the others; logistic IRT gives
   each MP one left–right score.
5. **Evaluation**: compare clusters with real parties (ARI, NMI) and check the result against MPs who later switched party.

## Data

Period: 1 Nov 2022 (election day) to 30 Sep 2025, which gives 1,377 voting sessions and about 233k individual ballots.

The 2022 legislature lasted until the election of 24 March 2026, but we stop before the 2025–26 parliamentary year because:
1. We keep three **complete** parliamentary years (2022–23, 2023–24, 2024–25). The 2025–26 year was cut short when the election was called in February 2026.
2. Votes just before an election are shaped by campaigning (symbolic motions, parties emphasising their differences), which could look like maverick behaviour.
3. MPs who switched party in late 2025 or early 2026 can serve as an **out-of-sample check**: we can test whether our methods flagged them as mavericks from earlier votes alone.

| Column | Description |
|---|---|
| `session` | Voting session id |
| `actor` | MP id |
| `vote` | 1 = For, 2 = Against, 3 = Absent, 4 = Abstain |
| `conclusion` | Official outcome text of the session |
| `passed` | Whether the proposal was adopted |
| `date` | Date of the meeting the session belongs to |
| `party` | The MP's party **on the date of the vote**, so party switches are preserved |
