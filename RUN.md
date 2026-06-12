# Step 1 — populate the Parquet store (idempotent, resumes from cursor)
cargo run --release --bin fetch-historical-data -- --data-dir ./data

# Step 2 — generate labelled feature CSV directly from the store
cargo run --release --bin collect-training-data -- \
    --data-dir ./data \
    --output-dir ./data/training \
    --label-window-secs 300

# Step 3 — walk-forward training + ONNX export
cd python && python train_direction_model.py \
    --training-dir ../data/training \
    --output model/direction_model.onnx \
    --scale short