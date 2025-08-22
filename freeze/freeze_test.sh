set -ex

# 1. no frozen
export FROZEN=0 FROZEN_LAYER=0 FROZEN_REQUIRE_GRAD=0 FROZEN_FILER_OPTIMIZER=0 FROZEN_NO_GRAD=0
python ./train_freeze.py > ./output/1.out 2>&1

# 2. frozen layer[0] with requres_grad
export FROZEN=1 FROZEN_LAYER=0 FROZEN_REQUIRE_GRAD=1 FROZEN_FILER_OPTIMIZER=0 FROZEN_NO_GRAD=0
python ./train_freeze.py > ./output/2.out 2>&1

# 3. frozen layer[1] with requres_grad
export FROZEN=1 FROZEN_LAYER=1 FROZEN_REQUIRE_GRAD=1 FROZEN_FILER_OPTIMIZER=0 FROZEN_NO_GRAD=0
python ./train_freeze.py > ./output/3.out 2>&1

# 4. frozen layer[0] with requres_grad & optimizer_filter
export FROZEN=1 FROZEN_LAYER=0 FROZEN_REQUIRE_GRAD=1 FROZEN_FILER_OPTIMIZER=1 FROZEN_NO_GRAD=0
python ./train_freeze.py > ./output/4.out 2>&1

# 5. frozen layer[1] with requres_grad & optimizer_filter
export FROZEN=1 FROZEN_LAYER=1 FROZEN_REQUIRE_GRAD=1 FROZEN_FILER_OPTIMIZER=1 FROZEN_NO_GRAD=0
python ./train_freeze.py > ./output/5.out 2>&1

# 6. frozen layer[0] with torch.no_grad
export FROZEN=1 FROZEN_LAYER=0 FROZEN_REQUIRE_GRAD=0 FROZEN_FILER_OPTIMIZER=0 FROZEN_NO_GRAD=1
python ./train_freeze.py > ./output/6.out 2>&1

# 7. frozen layer[1] with torch.no_grad
export FROZEN=1 FROZEN_LAYER=1 FROZEN_REQUIRE_GRAD=0 FROZEN_FILER_OPTIMIZER=0 FROZEN_NO_GRAD=1
python ./train_freeze.py > ./output/7.out 2>&1
