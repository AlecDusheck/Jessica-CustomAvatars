use std::collections::HashMap;
use lazy_static::lazy_static;

lazy_static! {
    pub static ref VERTEX_IDS: HashMap<String, HashMap<String, i64>> = {
        let mut vertex_ids = HashMap::new();

        let mut smplh = HashMap::new();
        smplh.insert("nose".to_string(), 332);
        smplh.insert("reye".to_string(), 6260);
        smplh.insert("leye".to_string(), 2800);
        smplh.insert("rear".to_string(), 4071);
        smplh.insert("lear".to_string(), 583);
        smplh.insert("rthumb".to_string(), 6191);
        smplh.insert("rindex".to_string(), 5782);
        smplh.insert("rmiddle".to_string(), 5905);
        smplh.insert("rring".to_string(), 6016);
        smplh.insert("rpinky".to_string(), 6133);
        smplh.insert("lthumb".to_string(), 2746);
        smplh.insert("lindex".to_string(), 2319);
        smplh.insert("lmiddle".to_string(), 2445);
        smplh.insert("lring".to_string(), 2556);
        smplh.insert("lpinky".to_string(), 2673);
        smplh.insert("LBigToe".to_string(), 3216);
        smplh.insert("LSmallToe".to_string(), 3226);
        smplh.insert("LHeel".to_string(), 3387);
        smplh.insert("RBigToe".to_string(), 6617);
        smplh.insert("RSmallToe".to_string(), 6624);
        smplh.insert("RHeel".to_string(), 6787);

        let mut smplx = HashMap::new();
        smplx.insert("nose".to_string(), 9120);
        smplx.insert("reye".to_string(), 9929);
        smplx.insert("leye".to_string(), 9448);
        smplx.insert("rear".to_string(), 616);
        smplx.insert("lear".to_string(), 6);
        smplx.insert("rthumb".to_string(), 8079);
        smplx.insert("rindex".to_string(), 7669);
        smplx.insert("rmiddle".to_string(), 7794);
        smplx.insert("rring".to_string(), 7905);
        smplx.insert("rpinky".to_string(), 8022);
        smplx.insert("lthumb".to_string(), 5361);
        smplx.insert("lindex".to_string(), 4933);
        smplx.insert("lmiddle".to_string(), 5058);
        smplx.insert("lring".to_string(), 5169);
        smplx.insert("lpinky".to_string(), 5286);
        smplx.insert("LBigToe".to_string(), 5770);
        smplx.insert("LSmallToe".to_string(), 5780);
        smplx.insert("LHeel".to_string(), 8846);
        smplx.insert("RBigToe".to_string(), 8463);
        smplx.insert("RSmallToe".to_string(), 8474);
        smplx.insert("RHeel".to_string(), 8635);

        let mut mano = HashMap::new();
        mano.insert("thumb".to_string(), 744);
        mano.insert("index".to_string(), 320);
        mano.insert("middle".to_string(), 443);
        mano.insert("ring".to_string(), 554);
        mano.insert("pinky".to_string(), 671);

        vertex_ids.insert("smplh".to_string(), smplh);
        vertex_ids.insert("smplx".to_string(), smplx);
        vertex_ids.insert("mano".to_string(), mano);

        vertex_ids
    };
}

lazy_static! {
    pub static ref JOINT_NAMES: Vec<&'static str> = vec![
        "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee", "spine2",
        "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot", "neck",
        "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow", "left_wrist", "right_wrist", "jaw",
        "left_eye_smplhf", "right_eye_smplhf", "left_index1", "left_index2", "left_index3",
        "left_middle1", "left_middle2", "left_middle3", "left_pinky1", "left_pinky2",
        "left_pinky3", "left_ring1", "left_ring2", "left_ring3", "left_thumb1",
        "left_thumb2", "left_thumb3", "right_index1", "right_index2", "right_index3",
        "right_middle1", "right_middle2", "right_middle3", "right_pinky1", "right_pinky2",
        "right_pinky3", "right_ring1", "right_ring2", "right_ring3", "right_thumb1",
        "right_thumb2", "right_thumb3", "nose", "right_eye", "left_eye", "right_ear",
        "left_ear", "left_big_toe", "left_small_toe", "left_heel", "right_big_toe",
        "right_small_toe", "right_heel", "left_thumb", "left_index", "left_middle",
        "left_ring", "left_pinky", "right_thumb", "right_index", "right_middle",
        "right_ring", "right_pinky", "right_eye_brow1", "right_eye_brow2",
        "right_eye_brow3", "right_eye_brow4", "right_eye_brow5", "left_eye_brow5",
        "left_eye_brow4", "left_eye_brow3", "left_eye_brow2", "left_eye_brow1",
        "nose1", "nose2", "nose3", "nose4", "right_nose_2", "right_nose_1",
        "nose_middle", "left_nose_1", "left_nose_2", "right_eye1", "right_eye2",
        "right_eye3", "right_eye4", "right_eye5", "right_eye6", "left_eye4",
        "left_eye3", "left_eye2", "left_eye1", "left_eye6", "left_eye5",
        "right_mouth_1", "right_mouth_2", "right_mouth_3", "mouth_top",
        "left_mouth_3", "left_mouth_2", "left_mouth_1", "left_mouth_5",
        "left_mouth_4", "mouth_bottom", "right_mouth_4", "right_mouth_5",
        "right_lip_1", "right_lip_2", "lip_top", "left_lip_2", "left_lip_1",
        "left_lip_3", "lip_bottom", "right_lip_3", "right_contour_1",
        "right_contour_2", "right_contour_3", "right_contour_4", "right_contour_5",
        "right_contour_6", "right_contour_7", "right_contour_8", "contour_middle",
        "left_contour_8", "left_contour_7", "left_contour_6", "left_contour_5",
        "left_contour_4", "left_contour_3", "left_contour_2", "left_contour_1",
    ];

    pub static ref SMPLH_JOINT_NAMES: Vec<&'static str> = vec![
        "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
        "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
        "neck", "left_collar", "right_collar", "head", "left_shoulder",
        "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
        "left_index1", "left_index2", "left_index3", "left_middle1", "left_middle2",
        "left_middle3", "left_pinky1", "left_pinky2", "left_pinky3", "left_ring1",
        "left_ring2", "left_ring3", "left_thumb1", "left_thumb2", "left_thumb3",
        "right_index1", "right_index2", "right_index3", "right_middle1",
        "right_middle2", "right_middle3", "right_pinky1", "right_pinky2",
        "right_pinky3", "right_ring1", "right_ring2", "right_ring3", "right_thumb1",
        "right_thumb2", "right_thumb3", "nose", "right_eye", "left_eye", "right_ear",
        "left_ear", "left_big_toe", "left_small_toe", "left_heel", "right_big_toe",
        "right_small_toe", "right_heel", "left_thumb", "left_index", "left_middle",
        "left_ring", "left_pinky", "right_thumb", "right_index", "right_middle",
        "right_ring", "right_pinky",
    ];
}