# #!/bin/bash

datasets=('lego')

for dt in "${datasets[@]}"
do
    python run_bionerf.py --config configs/$dt.txt --render_only --render_test --generate_samples
done
