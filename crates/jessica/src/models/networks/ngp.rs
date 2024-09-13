use jessica_tcnn_networks::cpp::TcnnModule;
use jessica_utils::module::ModuleMT;
use tch::{Kind, Tensor};

use crate::deformers::deformer::BoundingBox;

struct NeRFNGPNet {
    encoder: TcnnModule,
    color_net: TcnnModule,
    center: Tensor,
    scale: Tensor,
    bbox: Option<BoundingBox>,
}

impl NeRFNGPNet {
    pub fn new(center: Vec<f32>, scale: Vec<f32>) -> Self {
        let encoder = TcnnModule::new_encoder();
        let color_net = TcnnModule::new_color_net();

        let center = Tensor::from_slice(&center[..]).to_kind(Kind::Float);
        let scale = Tensor::from_slice(&scale[..]).to_kind(Kind::Float);

        Self {
            encoder,
            color_net,
            center,
            scale,
            bbox: None,
        }
    }

    pub fn initialize(&mut self, bbox: BoundingBox) {
        if self.bbox.is_some() {
            return;
        }

        let c = (&bbox.min_vert + &bbox.max_vert) / 2.0;
        let s = &bbox.max_vert - &bbox.min_vert;

        self.center = c;
        self.scale = s;
        self.bbox = Some(bbox);
    }
}

impl ModuleMT<Tensor, (Tensor, Tensor)> for NeRFNGPNet {
    fn forward_mt(&self, xs: Tensor, _train: bool) -> (Tensor, Tensor) {
        let xs = (xs - &self.center) / &self.scale + 0.5;
        let xs = xs.clamp(0.0, 1.0);

        let (_, x_encoded) = self.encoder.forward(&xs, &self.encoder.initial_params(42));
        let sigma = x_encoded.slice(1, 0, 1, 1);
        let color = self.color_net.forward(&x_encoded.slice(1, 1, 16, 1), &self.color_net.initial_params(42)).1.to_kind(Kind::Float);

        (color, sigma.to_kind(Kind::Float))
    }
}
