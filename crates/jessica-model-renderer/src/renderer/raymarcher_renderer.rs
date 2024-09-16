pub struct Raymarcher {
    MAX_SAMPLES: i32,
    MAX_BATCH_SIZE: i32,
    aabb: Tensor,
    density_grid_test: DensityGrid,
    density_grid_train_all: Vec<DensityGrid>,
    smpl_init: bool,
}