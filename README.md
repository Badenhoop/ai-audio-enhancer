# ai-audio-enhancer
An AI that turns low quality audio into high quality audio.

## Scraper

Scrapes a large set of cover songs from YouTube (~250GB).
The selected songs are compiled from the musicbrainz dataset and by filtering the most popular artists using the Spotify API.

## Denoising

A deep convolutional U-Net-style diffusion model trained on denoising songs.