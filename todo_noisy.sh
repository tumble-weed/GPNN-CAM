(
    realpath ~/bigfiles/other/results-librecam/ &
    mkdir ~/bigfiles/other/results-librecam/pascal-gpnn-vanilla &&
    mv ~/bigfiles/other/results-librecam/pascal-gpnn-* ~/bigfiles/other/results-librecam/pascal-gpnn-vanilla &&
)

python scripts/run_many_gpnn_may2.py