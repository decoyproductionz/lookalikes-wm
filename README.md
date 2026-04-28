# 2026 World Cup Lookalikes Project
Central repository for 2026 World Cup Lookalike project. Who's your football Doppelgänger?


## Technical Setup of Lookalikes Project

Face Embedding - uses pre-trained models (ArcFace, FaceNet, etc.) that already know what makes faces distinct. This is a more precise approach than building a CNN from scratch.

### Technical Work

Baseline model - CNN (Convolution Neural Network)

Improved model - Face Embedding Approach

## Selecting images

* Search term optimisation
* Search for "Player_name + player_country" + "headshot" <- Test this on lesser-known nation
* Include fallback scraper for Transfermarkt data, same can be done for SoccerWiki

### Notes from Friday presentation

* Include some method of cropping pictures to find only headshots of players
* Find cosine similarity inside group of players 
* Be mindful to only find pictures of actual players, no overlays