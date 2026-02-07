# Reinstallation Guide - Racing Gym RL (multi_car_racing)

Follow these steps to reinstall all packages for the multi_car_racing setup.

## Option 1: Automated Installation (Recommended)

```powershell
python install_all.py
```

## Option 2: Manual Step-by-Step Installation

### Step 1: Upgrade pip

```powershell
python -m pip install --upgrade pip
```

### Step 2: Install numpy (compatible with your Python version)

**For Python 3.11+:**
```powershell
pip install --prefer-binary "numpy>=1.23.0,<2.0"
```

**For Python 3.8-3.10:**
```powershell
pip install --prefer-binary "numpy>=1.22.0,<1.23.0"
```

### Step 3: Install core ML packages

```powershell
pip install --prefer-binary "gym==0.17.3" "stable-baselines3[extra]==1.8.0"
```

### Step 4: Install visualization packages

```powershell
pip install --prefer-binary "matplotlib>=3.7.0" "opencv-python>=4.8.0" "tensorboard>=2.13.0" "pyyaml>=6.0" "pyglet==1.5.27" "torch>=2.0.0"
```

### Step 5: Install multi_car_racing dependencies

```powershell
pip install "shapely==1.8.5.post1" "box2d-py==2.3.8"
```

### Step 6: Install multi_car_racing (without dependency checking)

```powershell
pip install git+https://github.com/igilitschenski/multi_car_racing.git --no-deps
```

### Step 7: Verify Installation

```powershell
python check_setup.py
```

Or test imports manually:

```powershell
python -c "import numpy, gym, gymnasium, shimmy, stable_baselines3, gym_multi_car_racing; print('All packages imported successfully!')"
```

## Troubleshooting

### If shapely fails to install:

- Install the pinned wheel:
  ```powershell
  pip install shapely==1.8.5.post1 --only-binary :all:
  ```

### If multi_car_racing import fails:

1. **Check if it's installed:**
   ```powershell
   pip show gym-multi-car-racing
   ```

2. **Reinstall without dependencies:**
   ```powershell
   pip uninstall gym-multi-car-racing -y
   pip install git+https://github.com/igilitschenski/multi_car_racing.git --no-deps
   ```

3. **Test import:**
   ```powershell
   python -c "import gym_multi_car_racing; print('multi_car_racing imported successfully')"
   ```

## Next Steps After Installation

1. **Verify setup:**
   ```powershell
   python check_setup.py
   ```

2. **Start training:**
   ```powershell
   python train.py --config config/multi_car_config.yaml
   ```

3. **Monitor training:**
   ```powershell
   tensorboard --logdir logs
   ```
