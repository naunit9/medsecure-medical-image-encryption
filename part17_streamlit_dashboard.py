# ============================================================
# MEDICAL IMAGE SECURITY PLATFORM
# COMPANY HANDOVER BUILD
# DEFAULT PAGE = ENCRYPT
# ============================================================

import io
import json
import time
import hashlib
from datetime import datetime
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn

from PIL import Image, ImageEnhance
from scipy.stats import skew, kurtosis
from skimage.feature import graycomatrix, graycoprops
from skimage.metrics import structural_similarity as ssim


# ============================================================
# OPTIONAL DICOM SUPPORT
# ============================================================

try:
    import pydicom
    DICOM_AVAILABLE = True

except ImportError:
    DICOM_AVAILABLE = False


# ============================================================
# STREAMLIT CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="MedSecure Adaptive Encryption",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_PATH = Path(__file__).resolve().parent

MODEL_FOLDER = (
    BASE_PATH
    / "Results"
    / "Deep_Learning_Model_Clean"
)

MODEL_PATH = (
    MODEL_FOLDER
    / "best_trained_model_clean.pth"
)

SCALER_PATH = (
    MODEL_FOLDER
    / "feature_scaler_clean.joblib"
)

CONFIG_PATH = (
    MODEL_FOLDER
    / "model_configuration_clean.json"
)


# ============================================================
# PRODUCT SETTINGS
# ============================================================

PRODUCT_NAME = "MedSecure"

PRODUCT_SUBTITLE = (
    "Adaptive Medical Image Encryption Platform"
)

CIPHER_VERSION = "MS-CHAOS-1.0"

IMAGE_SIZE = (
    256,
    256
)

BURN_IN = 500

FORWARD_CHAIN_SEED = 173

BACKWARD_CHAIN_SEED = 91


# ============================================================
# SESSION STATE
# ============================================================

if "encryption_result" not in st.session_state:
    st.session_state.encryption_result = None

if "upload_signature" not in st.session_state:
    st.session_state.upload_signature = None


# ============================================================
# UI STYLE
# ============================================================

st.markdown(
    """
<style>

.stApp {
    background:
        radial-gradient(
            circle at 8% 4%,
            rgba(34, 93, 255, 0.16),
            transparent 25%
        ),
        radial-gradient(
            circle at 92% 5%,
            rgba(123, 76, 255, 0.11),
            transparent 23%
        ),
        radial-gradient(
            circle at 75% 90%,
            rgba(0, 205, 255, 0.07),
            transparent 30%
        ),
        linear-gradient(
            145deg,
            #020710 0%,
            #07121f 48%,
            #030812 100%
        );

    color: #f5f8ff;
}


.block-container {
    max-width: 1580px;
    padding-top: 1rem;
    padding-bottom: 3rem;
}


#MainMenu {
    visibility: hidden;
}


footer {
    visibility: hidden;
}


header[data-testid="stHeader"] {
    background: transparent;
}


div[data-testid="stVerticalBlockBorderWrapper"] {
    background:
        linear-gradient(
            145deg,
            rgba(11, 27, 54, 0.88),
            rgba(6, 15, 31, 0.93)
        );

    border:
        1px solid rgba(69, 135, 255, 0.22) !important;

    border-radius: 20px;

    box-shadow:
        0 14px 40px rgba(0,0,0,0.18),
        inset 0 1px 0 rgba(255,255,255,0.025);
}


div[data-testid="stMetric"] {
    min-height: 102px;

    padding: 15px 16px;

    border-radius: 17px;

    border:
        1px solid rgba(75, 135, 255, 0.24);

    background:
        linear-gradient(
            145deg,
            rgba(12, 30, 61, 0.95),
            rgba(7, 17, 36, 0.97)
        );

    box-shadow:
        0 9px 28px rgba(0,0,0,0.15);
}


div[data-testid="stMetricLabel"] {
    color: #8ea8d1;
    font-size: 12px;
}


div[data-testid="stMetricValue"] {
    color: #f7faff;
    font-size: 25px;
    font-weight: 800;
}


div[role="radiogroup"] {
    background:
        rgba(6, 17, 36, 0.91);

    border:
        1px solid rgba(71, 128, 255, 0.23);

    border-radius: 16px;

    padding: 7px 9px;

    margin-bottom: 17px;
}


div[role="radiogroup"] label {
    border-radius: 10px;
    padding: 5px 10px;
}


div[data-testid="stFileUploader"] {
    background:
        rgba(7, 20, 43, 0.72);

    border:
        1px solid rgba(70, 139, 255, 0.24);

    border-radius: 18px;

    padding: 10px;
}


div[data-testid="stFileUploader"] section {
    border:
        1px dashed rgba(68, 190, 255, 0.57);

    border-radius: 14px;

    background:
        rgba(5, 17, 38, 0.65);
}


.stButton > button {
    width: 100%;
    min-height: 50px;

    border-radius: 14px;

    border:
        1px solid rgba(87, 153, 255, 0.58);

    background:
        linear-gradient(
            90deg,
            #1f61eb,
            #4759df,
            #0b98c6
        );

    color: white;

    font-weight: 800;

    box-shadow:
        0 0 24px rgba(56, 118, 255, 0.19);

    transition: 0.2s ease;
}


.stButton > button:hover {
    transform:
        translateY(-1px);

    border:
        1px solid #62e5ff;

    box-shadow:
        0 0 30px rgba(55, 193, 255, 0.26);
}


.stDownloadButton > button {
    width: 100%;

    min-height: 45px;

    border-radius: 12px;

    background:
        rgba(16, 39, 77, 0.90);

    color: white;

    border:
        1px solid rgba(79, 143, 255, 0.35);
}


div[data-testid="stDataFrame"] {
    border:
        1px solid rgba(75, 126, 255, 0.20);

    border-radius: 15px;

    overflow: hidden;
}


button[data-baseweb="tab"] {
    border-radius: 11px;

    background:
        rgba(10, 28, 57, 0.73);

    margin-right: 5px;

    padding-left: 18px;
    padding-right: 18px;
}


div[data-testid="stStatusWidget"] {
    border-radius: 16px;
}


div[data-testid="stAlert"] {
    border-radius: 14px;
}


h1 {
    font-weight: 850 !important;
    letter-spacing: -1.4px !important;
}


h2 {
    font-weight: 780 !important;
}


hr {
    border-color:
        rgba(83, 124, 212, 0.16);
}

</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODEL CONFIG
# ============================================================

@st.cache_resource
def load_configuration():

    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Configuration file not found:\n{CONFIG_PATH}"
        )

    with open(
        CONFIG_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


config = load_configuration()

FEATURE_COLUMNS = config[
    "feature_columns"
]

TARGET_COLUMNS = config[
    "target_columns"
]

PARAMETER_RANGES = config[
    "parameter_ranges"
]


# ============================================================
# MODEL
# ============================================================

class AdaptiveChaoticParameterModel(
    nn.Module
):

    def __init__(
        self,
        input_features=17,
        output_parameters=6
    ):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                input_features,
                128
            ),

            nn.BatchNorm1d(
                128
            ),

            nn.ReLU(),

            nn.Dropout(
                0.20
            ),

            nn.Linear(
                128,
                64
            ),

            nn.BatchNorm1d(
                64
            ),

            nn.ReLU(),

            nn.Dropout(
                0.15
            ),

            nn.Linear(
                64,
                32
            ),

            nn.ReLU(),

            nn.Linear(
                32,
                output_parameters
            ),

            nn.Sigmoid()
        )


    def forward(
        self,
        x
    ):

        return self.network(
            x
        )


# ============================================================
# LOAD MODEL AND SCALER
# ============================================================

@st.cache_resource
def load_model_and_scaler():

    device = torch.device(

        "cuda"

        if torch.cuda.is_available()

        else "cpu"
    )


    model = AdaptiveChaoticParameterModel(

        input_features=len(
            FEATURE_COLUMNS
        ),

        output_parameters=len(
            TARGET_COLUMNS
        )

    ).to(
        device
    )


    checkpoint = torch.load(

        MODEL_PATH,

        map_location=device,

        weights_only=False
    )


    model.load_state_dict(

        checkpoint[
            "model_state_dict"
        ]
    )


    model.eval()


    scaler = joblib.load(
        SCALER_PATH
    )


    return (
        model,
        scaler,
        device
    )


model, scaler, device = load_model_and_scaler()


# ============================================================
# NORMALIZE ARRAY
# ============================================================

def normalize_array_to_uint8(
    array
):

    array = np.asarray(
        array,
        dtype=np.float64
    )


    finite_mask = np.isfinite(
        array
    )


    if not finite_mask.any():

        return np.zeros(
            array.shape,
            dtype=np.uint8
        )


    valid = array[
        finite_mask
    ]


    lower = np.percentile(
        valid,
        1
    )


    upper = np.percentile(
        valid,
        99
    )


    if upper <= lower:

        lower = valid.min()
        upper = valid.max()


    if upper <= lower:

        return np.zeros(
            array.shape,
            dtype=np.uint8
        )


    clipped = np.clip(
        array,
        lower,
        upper
    )


    normalized = (

        (
            clipped
            -
            lower
        )

        /

        (
            upper
            -
            lower
        )

        *

        255.0
    )


    return np.clip(
        normalized,
        0,
        255
    ).astype(
        np.uint8
    )


# ============================================================
# DICOM LOADER
# ============================================================

def load_dicom_image(
    uploaded_file
):

    if not DICOM_AVAILABLE:

        raise RuntimeError(
            "DICOM support requires pydicom."
        )


    uploaded_file.seek(
        0
    )


    dataset = pydicom.dcmread(
        uploaded_file
    )


    pixels = dataset.pixel_array


    if pixels.ndim >= 3:

        if (
            pixels.ndim == 3
            and
            pixels.shape[-1] in [3, 4]
        ):

            rgb = normalize_array_to_uint8(
                pixels
            )


            if rgb.shape[-1] == 4:

                rgb = rgb[
                    :,
                    :,
                    :3
                ]


            pil_image = Image.fromarray(
                rgb
            ).convert(
                "L"
            )


        else:

            first_frame = pixels[
                0
            ]


            frame_uint8 = normalize_array_to_uint8(
                first_frame
            )


            pil_image = Image.fromarray(
                frame_uint8
            ).convert(
                "L"
            )


    else:

        pixel_array = pixels.astype(
            np.float64
        )


        slope = float(
            getattr(
                dataset,
                "RescaleSlope",
                1.0
            )
        )


        intercept = float(
            getattr(
                dataset,
                "RescaleIntercept",
                0.0
            )
        )


        pixel_array = (
            pixel_array
            *
            slope
            +
            intercept
        )


        image_uint8 = normalize_array_to_uint8(
            pixel_array
        )


        photometric = str(
            getattr(
                dataset,
                "PhotometricInterpretation",
                ""
            )
        ).upper()


        if photometric == "MONOCHROME1":

            image_uint8 = (
                255
                -
                image_uint8
            )


        pil_image = Image.fromarray(
            image_uint8
        ).convert(
            "L"
        )


    metadata = {

        "format":
            "DICOM",

        "rows":
            getattr(
                dataset,
                "Rows",
                None
            ),

        "columns":
            getattr(
                dataset,
                "Columns",
                None
            ),

        "modality":
            getattr(
                dataset,
                "Modality",
                "Unknown"
            ),

        "photometric_interpretation":
            getattr(
                dataset,
                "PhotometricInterpretation",
                "Unknown"
            ),

        "frames":
            getattr(
                dataset,
                "NumberOfFrames",
                1
            )
    }


    return (
        pil_image,
        metadata
    )


# ============================================================
# GENERAL IMAGE LOADER
# ============================================================

def load_medical_image(
    uploaded_file
):

    extension = Path(
        uploaded_file.name
    ).suffix.lower()


    if extension == ".dcm":

        return load_dicom_image(
            uploaded_file
        )


    uploaded_file.seek(
        0
    )


    image = Image.open(
        uploaded_file
    )


    metadata = {

        "format":
            image.format
            or
            extension.replace(
                ".",
                ""
            ).upper(),

        "width":
            image.size[
                0
            ],

        "height":
            image.size[
                1
            ],

        "mode":
            image.mode
    }


    return (
        image,
        metadata
    )


# ============================================================
# PREPROCESS
# ============================================================

def preprocess_image(
    image
):

    image = image.convert(
        "L"
    )


    image = image.resize(
        IMAGE_SIZE
    )


    image = ImageEnhance.Contrast(
        image
    ).enhance(
        1.1
    )


    return np.array(
        image,
        dtype=np.uint8
    )


# ============================================================
# ENTROPY
# ============================================================

def calculate_entropy(
    image
):

    histogram = np.bincount(

        image.flatten(),

        minlength=256
    )


    probabilities = (

        histogram[
            histogram > 0
        ]

        /

        image.size
    )


    return float(

        -np.sum(

            probabilities

            *

            np.log2(
                probabilities
            )
        )
    )


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(
    image
):

    flat = image.flatten().astype(
        np.float64
    )


    histogram = np.bincount(

        image.flatten(),

        minlength=256

    ).astype(
        np.float64
    )


    probabilities = (

        histogram

        /

        histogram.sum()
    )


    edges = cv2.Canny(
        image,
        100,
        200
    )


    quantized = (
        image
        //
        8
    ).astype(
        np.uint8
    )


    glcm = graycomatrix(

        quantized,

        distances=[
            1
        ],

        angles=[
            0,
            np.pi / 4,
            np.pi / 2,
            3 * np.pi / 4
        ],

        levels=32,

        symmetric=True,

        normed=True
    )


    return {

        "mean":
            float(
                np.mean(
                    flat
                )
            ),

        "std":
            float(
                np.std(
                    flat
                )
            ),

        "variance":
            float(
                np.var(
                    flat
                )
            ),

        "median":
            float(
                np.median(
                    flat
                )
            ),

        "min_intensity":
            float(
                np.min(
                    flat
                )
            ),

        "max_intensity":
            float(
                np.max(
                    flat
                )
            ),

        "skewness":
            float(
                skew(
                    flat
                )
            ),

        "kurtosis":
            float(
                kurtosis(
                    flat
                )
            ),

        "entropy":
            calculate_entropy(
                image
            ),

        "histogram_energy":
            float(
                np.sum(
                    probabilities ** 2
                )
            ),

        "edge_density":
            float(
                np.mean(
                    edges > 0
                )
            ),

        "glcm_contrast":
            float(
                np.mean(
                    graycoprops(
                        glcm,
                        "contrast"
                    )
                )
            ),

        "glcm_dissimilarity":
            float(
                np.mean(
                    graycoprops(
                        glcm,
                        "dissimilarity"
                    )
                )
            ),

        "glcm_homogeneity":
            float(
                np.mean(
                    graycoprops(
                        glcm,
                        "homogeneity"
                    )
                )
            ),

        "glcm_energy":
            float(
                np.mean(
                    graycoprops(
                        glcm,
                        "energy"
                    )
                )
            ),

        "glcm_correlation":
            float(
                np.mean(
                    graycoprops(
                        glcm,
                        "correlation"
                    )
                )
            ),

        "glcm_asm":
            float(
                np.mean(
                    graycoprops(
                        glcm,
                        "ASM"
                    )
                )
            )
    }


# ============================================================
# PARAMETER PREDICTION
# ============================================================

def predict_parameters(
    features
):

    vector = np.array(

        [

            features[
                column
            ]

            for column
            in FEATURE_COLUMNS
        ],

        dtype=np.float64

    ).reshape(
        1,
        -1
    )


    vector_scaled = scaler.transform(
        vector
    ).astype(
        np.float32
    )


    tensor = torch.tensor(

        vector_scaled,

        dtype=torch.float32,

        device=device
    )


    with torch.no_grad():

        normalized = (

            model(
                tensor
            )

            .cpu()

            .numpy()[0]
        )


    parameters = {}


    for index, parameter_name in enumerate(
        TARGET_COLUMNS
    ):

        minimum, maximum = PARAMETER_RANGES[
            parameter_name
        ]


        minimum = float(
            minimum
        )


        maximum = float(
            maximum
        )


        parameters[
            parameter_name
        ] = float(

            minimum

            +

            normalized[
                index
            ]

            *

            (
                maximum
                -
                minimum
            )
        )


    return parameters


# ============================================================
# CHAOTIC MAPS
# ============================================================

def logistic_map(
    x0,
    r,
    length
):

    sequence = np.empty(

        length
        +
        BURN_IN,

        dtype=np.float64
    )


    x = float(
        x0
    )


    for i in range(
        length
        +
        BURN_IN
    ):

        x = (
            r
            *
            x
            *
            (
                1.0
                -
                x
            )
        )


        sequence[
            i
        ] = x


    return sequence[
        BURN_IN:
    ]


def tent_map(
    x0,
    mu,
    length
):

    sequence = np.empty(

        length
        +
        BURN_IN,

        dtype=np.float64
    )


    x = float(
        x0
    )


    for i in range(
        length
        +
        BURN_IN
    ):

        if x < 0.5:

            x = (
                mu
                *
                x
            )

        else:

            x = (
                mu
                *
                (
                    1.0
                    -
                    x
                )
            )


        sequence[
            i
        ] = x


    return sequence[
        BURN_IN:
    ]


def chaotic_to_uint8(
    sequence
):

    scaled = np.floor(

        sequence
        *
        1e14

    ).astype(
        np.uint64
    )


    return (
        scaled
        %
        256
    ).astype(
        np.uint8
    )


# ============================================================
# BIT ROTATION
# ============================================================

def rotate_left_8(
    value,
    shift
):

    value = int(
        value
    )


    shift = (
        int(
            shift
        )
        &
        7
    )


    if shift == 0:

        return value


    return (

        (
            value
            <<
            shift
        )

        |

        (
            value
            >>
            (
                8
                -
                shift
            )
        )

    ) & 0xFF


def rotate_right_8(
    value,
    shift
):

    value = int(
        value
    )


    shift = (
        int(
            shift
        )
        &
        7
    )


    if shift == 0:

        return value


    return (

        (
            value
            >>
            shift
        )

        |

        (
            value
            <<
            (
                8
                -
                shift
            )
        )

    ) & 0xFF


# ============================================================
# FORWARD DIFFUSION
# ============================================================

def forward_diffusion(
    flat,
    key
):

    output = np.empty_like(
        flat
    )


    previous_cipher = (
        FORWARD_CHAIN_SEED
    )


    for index in range(
        len(
            flat
        )
    ):

        mixed = (

            int(
                flat[
                    index
                ]
            )

            ^

            int(
                key[
                    index
                ]
            )
        )


        value = (
            mixed
            +
            previous_cipher
        ) & 0xFF


        shift = (
            int(
                key[
                    index
                ]
            )
            &
            7
        )


        cipher_value = rotate_left_8(
            value,
            shift
        )


        output[
            index
        ] = cipher_value


        previous_cipher = cipher_value


    return output


# ============================================================
# INVERSE FORWARD
# ============================================================

def inverse_forward_diffusion(
    flat,
    key
):

    output = np.empty_like(
        flat
    )


    previous_cipher = (
        FORWARD_CHAIN_SEED
    )


    for index in range(
        len(
            flat
        )
    ):

        cipher_value = int(
            flat[
                index
            ]
        )


        shift = (
            int(
                key[
                    index
                ]
            )
            &
            7
        )


        value = rotate_right_8(
            cipher_value,
            shift
        )


        mixed = (
            value
            -
            previous_cipher
        ) & 0xFF


        output[
            index
        ] = (
            mixed
            ^
            int(
                key[
                    index
                ]
            )
        )


        previous_cipher = cipher_value


    return output


# ============================================================
# BACKWARD DIFFUSION
# ============================================================

def backward_diffusion(
    flat,
    key
):

    output = np.empty_like(
        flat
    )


    next_cipher = (
        BACKWARD_CHAIN_SEED
    )


    for index in range(

        len(
            flat
        ) - 1,

        -1,

        -1
    ):

        mixed = (

            int(
                flat[
                    index
                ]
            )

            ^

            int(
                key[
                    index
                ]
            )
        )


        value = (
            mixed
            +
            next_cipher
        ) & 0xFF


        shift = (
            int(
                key[
                    index
                ]
            )
            &
            7
        )


        cipher_value = rotate_left_8(
            value,
            shift
        )


        output[
            index
        ] = cipher_value


        next_cipher = cipher_value


    return output


# ============================================================
# INVERSE BACKWARD
# ============================================================

def inverse_backward_diffusion(
    flat,
    key
):

    output = np.empty_like(
        flat
    )


    next_cipher = (
        BACKWARD_CHAIN_SEED
    )


    for index in range(

        len(
            flat
        ) - 1,

        -1,

        -1
    ):

        cipher_value = int(
            flat[
                index
            ]
        )


        shift = (
            int(
                key[
                    index
                ]
            )
            &
            7
        )


        value = rotate_right_8(
            cipher_value,
            shift
        )


        mixed = (
            value
            -
            next_cipher
        ) & 0xFF


        output[
            index
        ] = (
            mixed
            ^
            int(
                key[
                    index
                ]
            )
        )


        next_cipher = cipher_value


    return output


# ============================================================
# GENERATE CIPHER DATA
# ============================================================

def generate_encryption_data(
    shape,
    parameters
):

    total_pixels = (
        shape[
            0
        ]
        *
        shape[
            1
        ]
    )


    permutation_sequence = logistic_map(

        parameters[
            "permutation_x0"
        ],

        parameters[
            "permutation_r"
        ],

        total_pixels
    )


    permutation_index = np.argsort(

        permutation_sequence,

        kind="stable"
    )


    forward_sequence = tent_map(

        parameters[
            "forward_x0"
        ],

        parameters[
            "forward_mu"
        ],

        total_pixels
    )


    forward_key = chaotic_to_uint8(
        forward_sequence
    )


    backward_sequence = logistic_map(

        parameters[
            "backward_x0"
        ],

        parameters[
            "backward_r"
        ],

        total_pixels
    )


    backward_key = chaotic_to_uint8(
        backward_sequence
    )


    return (
        permutation_index,
        forward_key,
        backward_key
    )


# ============================================================
# ENCRYPT
# ============================================================

def encrypt_image(
    image,
    parameters
):

    (
        permutation_index,
        forward_key,
        backward_key

    ) = generate_encryption_data(
        image.shape,
        parameters
    )


    flat = image.flatten()


    permuted = flat[
        permutation_index
    ]


    forward_result = forward_diffusion(
        permuted,
        forward_key
    )


    encrypted = backward_diffusion(
        forward_result,
        backward_key
    )


    return encrypted.reshape(
        image.shape
    )


# ============================================================
# DECRYPT
# ============================================================

def decrypt_image(
    encrypted,
    parameters
):

    (
        permutation_index,
        forward_key,
        backward_key

    ) = generate_encryption_data(
        encrypted.shape,
        parameters
    )


    flat = encrypted.flatten()


    backward_removed = inverse_backward_diffusion(
        flat,
        backward_key
    )


    forward_removed = inverse_forward_diffusion(
        backward_removed,
        forward_key
    )


    recovered = np.empty_like(
        forward_removed
    )


    recovered[
        permutation_index
    ] = forward_removed


    return recovered.reshape(
        encrypted.shape
    )


# ============================================================
# METRICS
# ============================================================

def correlation(
    first,
    second
):

    first = first.astype(
        np.float64
    ).flatten()


    second = second.astype(
        np.float64
    ).flatten()


    if (
        np.std(
            first
        ) == 0

        or

        np.std(
            second
        ) == 0
    ):

        return 0.0


    return float(
        np.corrcoef(
            first,
            second
        )[0, 1]
    )


def image_correlations(
    image
):

    return (

        correlation(
            image[:, :-1],
            image[:, 1:]
        ),

        correlation(
            image[:-1, :],
            image[1:, :]
        ),

        correlation(
            image[:-1, :-1],
            image[1:, 1:]
        )
    )


def calculate_npcr(
    first,
    second
):

    return float(

        np.mean(
            first
            !=
            second
        )

        *
        100.0
    )


def calculate_uaci(
    first,
    second
):

    return float(

        np.mean(

            np.abs(

                first.astype(
                    np.float64
                )

                -

                second.astype(
                    np.float64
                )
            )

            /

            255.0
        )

        *
        100.0
    )


def calculate_mse(
    first,
    second
):

    difference = (
        first.astype(
            np.float64
        )
        -
        second.astype(
            np.float64
        )
    )


    return float(
        np.mean(
            difference ** 2
        )
    )


def calculate_psnr(
    first,
    second
):

    mse_value = calculate_mse(
        first,
        second
    )


    if mse_value == 0:

        return float(
            "inf"
        )


    return float(

        10

        *

        np.log10(

            (
                255.0 ** 2
            )

            /

            mse_value
        )
    )


# ============================================================
# DIFFERENTIAL TEST
# ============================================================

def modify_center_pixel(
    image
):

    modified = image.copy()


    y = (
        image.shape[
            0
        ]
        //
        2
    )


    x = (
        image.shape[
            1
        ]
        //
        2
    )


    value = int(
        modified[
            y,
            x
        ]
    )


    if value < 255:

        modified[
            y,
            x
        ] = value + 1

    else:

        modified[
            y,
            x
        ] = value - 1


    return modified


# ============================================================
# HASH
# ============================================================

def sha256_image(
    image
):

    return hashlib.sha256(

        image.astype(
            np.uint8
        ).tobytes()

    ).hexdigest()


# ============================================================
# IMAGE TO PNG
# ============================================================

def image_to_png_bytes(
    image
):

    buffer = io.BytesIO()


    Image.fromarray(

        image.astype(
            np.uint8
        )

    ).save(
        buffer,
        format="PNG"
    )


    return buffer.getvalue()


# ============================================================
# KEY PACKAGE
# ============================================================

def create_key_package(
    parameters,
    processed_image,
    encrypted_image
):

    return {

        "product":
            PRODUCT_NAME,

        "cipher_version":
            CIPHER_VERSION,

        "created_utc":
            datetime.utcnow().isoformat()
            +
            "Z",

        "image_width":
            int(
                encrypted_image.shape[
                    1
                ]
            ),

        "image_height":
            int(
                encrypted_image.shape[
                    0
                ]
            ),

        "burn_in":
            BURN_IN,

        "permutation_sort":
            "stable",

        "forward_chain_seed":
            FORWARD_CHAIN_SEED,

        "backward_chain_seed":
            BACKWARD_CHAIN_SEED,

        "parameters_hex": {

            name:
                float(
                    value
                ).hex()

            for name, value
            in parameters.items()
        },

        "parameters_decimal": {

            name:
                repr(
                    float(
                        value
                    )
                )

            for name, value
            in parameters.items()
        },

        "processed_image_sha256":
            sha256_image(
                processed_image
            ),

        "encrypted_image_sha256":
            sha256_image(
                encrypted_image
            )
    }


# ============================================================
# READ PARAMETERS FROM KEY FILE
# ============================================================

def parameters_from_key_package(
    key_package
):

    if (
        key_package.get(
            "cipher_version"
        )
        !=
        CIPHER_VERSION
    ):

        raise ValueError(
            "Unsupported cipher version."
        )


    hex_parameters = key_package.get(
        "parameters_hex"
    )


    if not isinstance(
        hex_parameters,
        dict
    ):

        raise ValueError(
            "Key file does not contain exact parameter information."
        )


    parameters = {}


    for parameter_name in TARGET_COLUMNS:

        if parameter_name not in hex_parameters:

            raise ValueError(
                f"Missing parameter: {parameter_name}"
            )


        parameters[
            parameter_name
        ] = float.fromhex(

            hex_parameters[
                parameter_name
            ]
        )


    return parameters


# ============================================================
# SECURITY REPORT
# ============================================================

def create_security_report(
    filename,
    source_metadata,
    parameters,
    metrics,
    timings
):

    return {

        "timestamp":
            datetime.now().isoformat(),

        "product":
            PRODUCT_NAME,

        "cipher_version":
            CIPHER_VERSION,

        "source_filename":
            filename,

        "source_metadata":
            source_metadata,

        "processing_resolution":
            "256x256 grayscale",

        "parameters": {

            key:
                float(
                    value
                )

            for key, value
            in parameters.items()
        },

        "metrics":
            metrics,

        "timings_seconds":
            timings
    }


# ============================================================
# HEADER
# ============================================================

header_left, header_center, header_right = st.columns(
    [
        2.5,
        1.5,
        1
    ]
)


with header_left:

    st.title(
        "🔐 MedSecure"
    )

    st.caption(
        "Adaptive Medical Image Encryption Platform"
    )


with header_center:

    h1, h2, h3 = st.columns(
        3
    )


    h1.metric(
        "Cipher",
        "Adaptive"
    )


    h2.metric(
        "Recovery",
        "Lossless"
    )


    h3.metric(
        "Input",
        "Medical"
    )


with header_right:

    st.success(
        "● ENGINE READY"
    )

    st.caption(
        f"Runtime: {device}"
    )


# ============================================================
# NAVIGATION
# DEFAULT = ENCRYPT
# ============================================================

page = st.radio(

    "Navigation",

    [
        "🔒 Encrypt",
        "🔓 Decrypt",
        "🔬 Image Analysis"
    ],

    horizontal=True,

    index=0,

    label_visibility="collapsed"
)


# ============================================================
# ENCRYPT PAGE
# ============================================================

if page == "🔒 Encrypt":

    st.title(
        "Encrypt Medical Image"
    )


    st.caption(
        "Upload a medical image, generate adaptive parameters, encrypt it and export the protected image with its matching key."
    )


    uploaded_file = st.file_uploader(

        "Upload medical image",

        type=[
            "png",
            "jpg",
            "jpeg",
            "bmp",
            "tif",
            "tiff",
            "dcm"
        ],

        key="encrypt_upload"
    )


    if uploaded_file is None:

        st.info(
            "Select a supported medical image to begin."
        )


    else:

        try:

            (
                source_image,
                source_metadata

            ) = load_medical_image(
                uploaded_file
            )


        except Exception as error:

            st.error(
                f"Unable to load image: {error}"
            )

            st.stop()


        signature = (
            uploaded_file.name,
            uploaded_file.size
        )


        if (
            st.session_state.upload_signature
            !=
            signature
        ):

            st.session_state.upload_signature = (
                signature
            )

            st.session_state.encryption_result = None


        left, right = st.columns(
            [
                1,
                1.55
            ]
        )


        with left:

            with st.container(
                border=True
            ):

                st.subheader(
                    "Source Image"
                )


                st.image(
                    source_image,
                    use_container_width=True
                )


                st.write(
                    f"**File:** `{uploaded_file.name}`"
                )


                st.write(
                    f"**Format:** {source_metadata.get('format', 'Unknown')}"
                )


                if (
                    source_metadata.get(
                        "modality"
                    )
                    is not None
                ):

                    st.write(
                        f"**DICOM Modality:** {source_metadata.get('modality')}"
                    )


        with right:

            with st.container(
                border=True
            ):

                st.subheader(
                    "Encryption Configuration"
                )


                e1, e2 = st.columns(
                    2
                )


                e1.metric(
                    "Processing",
                    "256 × 256"
                )


                e2.metric(
                    "Color Mode",
                    "Grayscale"
                )


                e3, e4 = st.columns(
                    2
                )


                e3.metric(
                    "Parameter Mode",
                    "Adaptive"
                )


                e4.metric(
                    "Cipher",
                    CIPHER_VERSION
                )


                st.write(
                    "The uploaded image is standardized before adaptive encryption."
                )


                encrypt_button = st.button(

                    "🔐 ENCRYPT IMAGE",

                    type="primary",

                    use_container_width=True
                )


        if encrypt_button:

            with st.status(
                "Encrypting medical image...",
                expanded=True
            ) as status:

                st.write(
                    "Standardizing image..."
                )


                processed_image = preprocess_image(
                    source_image
                )


                st.write(
                    "Analyzing image characteristics..."
                )


                feature_start = time.perf_counter()


                features = extract_features(
                    processed_image
                )


                feature_time = (
                    time.perf_counter()
                    -
                    feature_start
                )


                st.write(
                    "Generating adaptive encryption parameters..."
                )


                prediction_start = time.perf_counter()


                parameters = predict_parameters(
                    features
                )


                prediction_time = (
                    time.perf_counter()
                    -
                    prediction_start
                )


                st.write(
                    "Running chaotic encryption engine..."
                )


                encryption_start = time.perf_counter()


                encrypted_image = encrypt_image(
                    processed_image,
                    parameters
                )


                encryption_time = (
                    time.perf_counter()
                    -
                    encryption_start
                )


                st.write(
                    "Verifying reversible recovery..."
                )


                recovered_check = decrypt_image(
                    encrypted_image,
                    parameters
                )


                exact_recovery = np.array_equal(
                    processed_image,
                    recovered_check
                )


                st.write(
                    "Calculating security metrics..."
                )


                changed_image = modify_center_pixel(
                    processed_image
                )


                changed_cipher = encrypt_image(
                    changed_image,
                    parameters
                )


                cipher_entropy = calculate_entropy(
                    encrypted_image
                )


                (
                    horizontal_corr,
                    vertical_corr,
                    diagonal_corr

                ) = image_correlations(
                    encrypted_image
                )


                npcr_value = calculate_npcr(
                    encrypted_image,
                    changed_cipher
                )


                uaci_value = calculate_uaci(
                    encrypted_image,
                    changed_cipher
                )


                key_package = create_key_package(
                    parameters,
                    processed_image,
                    encrypted_image
                )


                metrics = {

                    "cipher_entropy":
                        cipher_entropy,

                    "npcr_percent":
                        npcr_value,

                    "uaci_percent":
                        uaci_value,

                    "horizontal_correlation":
                        horizontal_corr,

                    "vertical_correlation":
                        vertical_corr,

                    "diagonal_correlation":
                        diagonal_corr,

                    "exact_recovery":
                        bool(
                            exact_recovery
                        )
                }


                timings = {

                    "feature_extraction":
                        feature_time,

                    "parameter_generation":
                        prediction_time,

                    "encryption":
                        encryption_time
                }


                security_report = create_security_report(

                    uploaded_file.name,

                    source_metadata,

                    parameters,

                    metrics,

                    timings
                )


                st.session_state.encryption_result = {

                    "processed":
                        processed_image,

                    "encrypted":
                        encrypted_image,

                    "recovered_check":
                        recovered_check,

                    "features":
                        features,

                    "parameters":
                        parameters,

                    "key_package":
                        key_package,

                    "report":
                        security_report,

                    "metrics":
                        metrics,

                    "timings":
                        timings
                }


                status.update(

                    label=
                        "Encryption completed successfully",

                    state=
                        "complete",

                    expanded=
                        False
                )


        if (
            st.session_state.encryption_result
            is not None
        ):

            result = (
                st.session_state.encryption_result
            )


            st.divider()


            st.subheader(
                "Encryption Result"
            )


            i1, i2, i3 = st.columns(
                3
            )


            with i1:

                with st.container(
                    border=True
                ):

                    st.markdown(
                        "#### Standardized Source"
                    )


                    st.image(
                        result[
                            "processed"
                        ],
                        use_container_width=True,
                        clamp=True
                    )


            with i2:

                with st.container(
                    border=True
                ):

                    st.markdown(
                        "#### Encrypted Image"
                    )


                    st.image(
                        result[
                            "encrypted"
                        ],
                        use_container_width=True,
                        clamp=True
                    )


            with i3:

                with st.container(
                    border=True
                ):

                    st.markdown(
                        "#### Recovery Verification"
                    )


                    st.image(
                        result[
                            "recovered_check"
                        ],
                        use_container_width=True,
                        clamp=True
                    )


            st.write("")


            m1, m2, m3, m4 = st.columns(
                4
            )


            m1.metric(
                "Cipher Entropy",
                f"{result['metrics']['cipher_entropy']:.6f}"
            )


            m2.metric(
                "NPCR",
                f"{result['metrics']['npcr_percent']:.4f}%"
            )


            m3.metric(
                "UACI",
                f"{result['metrics']['uaci_percent']:.4f}%"
            )


            m4.metric(
                "Recovery",
                "VERIFIED"
                if result[
                    "metrics"
                ][
                    "exact_recovery"
                ]
                else "FAILED"
            )


            c1, c2, c3 = st.columns(
                3
            )


            c1.metric(
                "Horizontal Corr.",
                f"{result['metrics']['horizontal_correlation']:.6f}"
            )


            c2.metric(
                "Vertical Corr.",
                f"{result['metrics']['vertical_correlation']:.6f}"
            )


            c3.metric(
                "Diagonal Corr.",
                f"{result['metrics']['diagonal_correlation']:.6f}"
            )


            (
                parameter_tab,
                feature_tab,
                export_tab

            ) = st.tabs(
                [
                    "Adaptive Parameters",
                    "Image Analysis",
                    "Export"
                ]
            )


            with parameter_tab:

                parameter_dataframe = pd.DataFrame(

                    {

                        "Parameter":
                            list(
                                result[
                                    "parameters"
                                ].keys()
                            ),

                        "Value":
                            list(
                                result[
                                    "parameters"
                                ].values()
                            ),

                        "Exact Representation":
                            [

                                float(
                                    value
                                ).hex()

                                for value
                                in result[
                                    "parameters"
                                ].values()
                            ]
                    }
                )


                st.dataframe(
                    parameter_dataframe,
                    hide_index=True,
                    use_container_width=True
                )


            with feature_tab:

                feature_dataframe = pd.DataFrame(

                    {

                        "Feature":
                            FEATURE_COLUMNS,

                        "Value":
                            [

                                result[
                                    "features"
                                ][feature]

                                for feature
                                in FEATURE_COLUMNS
                            ]
                    }
                )


                st.dataframe(
                    feature_dataframe,
                    hide_index=True,
                    use_container_width=True
                )


            with export_tab:

                st.success(
                    "Download the encrypted image and key file for later independent decryption."
                )


                encrypted_bytes = image_to_png_bytes(
                    result[
                        "encrypted"
                    ]
                )


                key_json = json.dumps(
                    result[
                        "key_package"
                    ],
                    indent=4
                )


                report_json = json.dumps(
                    result[
                        "report"
                    ],
                    indent=4,
                    default=str
                )


                d1, d2, d3 = st.columns(
                    3
                )


                with d1:

                    st.download_button(

                        "⬇ Encrypted Image",

                        data=
                            encrypted_bytes,

                        file_name=
                            "encrypted_medical_image.png",

                        mime=
                            "image/png",

                        use_container_width=True
                    )


                with d2:

                    st.download_button(

                        "🔑 Encryption Key",

                        data=
                            key_json,

                        file_name=
                            "medical_image_key.json",

                        mime=
                            "application/json",

                        use_container_width=True
                    )


                with d3:

                    st.download_button(

                        "📄 Security Report",

                        data=
                            report_json,

                        file_name=
                            "medical_image_security_report.json",

                        mime=
                            "application/json",

                        use_container_width=True
                    )


# ============================================================
# DECRYPT PAGE
# ============================================================

elif page == "🔓 Decrypt":

    st.title(
        "Decrypt Medical Image"
    )


    st.caption(
        "Upload the encrypted PNG and its matching encryption-key JSON file."
    )


    encrypted_upload = st.file_uploader(

        "Encrypted image",

        type=[
            "png"
        ],

        key="decrypt_image"
    )


    key_upload = st.file_uploader(

        "Encryption key file",

        type=[
            "json"
        ],

        key="decrypt_key"
    )


    if (
        encrypted_upload is None

        or

        key_upload is None
    ):

        st.info(
            "Upload both files to continue."
        )


    else:

        try:

            encrypted_pil = Image.open(
                encrypted_upload
            ).convert(
                "L"
            )


            encrypted_array = np.array(
                encrypted_pil,
                dtype=np.uint8
            )


            key_upload.seek(
                0
            )


            key_package = json.load(
                key_upload
            )


            parameters = parameters_from_key_package(
                key_package
            )


            expected_shape = (

                int(
                    key_package[
                        "image_height"
                    ]
                ),

                int(
                    key_package[
                        "image_width"
                    ]
                )
            )


            if (
                encrypted_array.shape
                !=
                expected_shape
            ):

                st.error(
                    "Encrypted image dimensions do not match the key file."
                )

                st.stop()


            expected_cipher_hash = key_package.get(
                "encrypted_image_sha256"
            )


            actual_cipher_hash = sha256_image(
                encrypted_array
            )


            hash_match = (
                expected_cipher_hash
                ==
                actual_cipher_hash
            )


            v1, v2, v3 = st.columns(
                3
            )


            v1.metric(
                "Cipher Version",
                key_package.get(
                    "cipher_version",
                    "Unknown"
                )
            )


            v2.metric(
                "Image Size",
                f"{encrypted_array.shape[1]} × {encrypted_array.shape[0]}"
            )


            v3.metric(
                "Integrity",
                "VALID"
                if hash_match
                else "CHANGED"
            )


            if not hash_match:

                st.warning(
                    "The encrypted image does not match the hash stored in the key file."
                )


            decrypt_button = st.button(

                "🔓 DECRYPT IMAGE",

                type="primary",

                use_container_width=True
            )


            if decrypt_button:

                with st.status(
                    "Decrypting image...",
                    expanded=True
                ) as status:

                    start_time = time.perf_counter()


                    recovered = decrypt_image(
                        encrypted_array,
                        parameters
                    )


                    decryption_time = (
                        time.perf_counter()
                        -
                        start_time
                    )


                    recovered_hash = sha256_image(
                        recovered
                    )


                    expected_recovered_hash = key_package.get(
                        "processed_image_sha256"
                    )


                    recovery_verified = (
                        recovered_hash
                        ==
                        expected_recovered_hash
                    )


                    status.update(

                        label=
                            "Decryption completed",

                        state=
                            "complete",

                        expanded=
                            False
                    )


                d1, d2 = st.columns(
                    2
                )


                with d1:

                    with st.container(
                        border=True
                    ):

                        st.markdown(
                            "#### Encrypted Image"
                        )


                        st.image(
                            encrypted_array,
                            use_container_width=True,
                            clamp=True
                        )


                with d2:

                    with st.container(
                        border=True
                    ):

                        st.markdown(
                            "#### Recovered Image"
                        )


                        st.image(
                            recovered,
                            use_container_width=True,
                            clamp=True
                        )


                r1, r2 = st.columns(
                    2
                )


                r1.metric(
                    "Decryption Time",
                    f"{decryption_time:.4f}s"
                )


                r2.metric(
                    "Recovery Integrity",
                    "VERIFIED"
                    if recovery_verified
                    else "NOT VERIFIED"
                )


                if recovery_verified:

                    st.success(
                        "Recovered image integrity verified."
                    )

                else:

                    st.warning(
                        "Recovered image hash does not match the original protected image."
                    )


                st.download_button(

                    "⬇ Download Recovered Image",

                    data=
                        image_to_png_bytes(
                            recovered
                        ),

                    file_name=
                        "recovered_medical_image.png",

                    mime=
                        "image/png",

                    use_container_width=True
                )


        except Exception as error:

            st.error(
                f"Unable to decrypt package: {error}"
            )


# ============================================================
# IMAGE ANALYSIS PAGE
# ============================================================

elif page == "🔬 Image Analysis":

    st.title(
        "Medical Image Analysis"
    )


    st.caption(
        "Inspect image characteristics and the adaptive parameters generated for that image."
    )


    analysis_upload = st.file_uploader(

        "Upload medical image",

        type=[
            "png",
            "jpg",
            "jpeg",
            "bmp",
            "tif",
            "tiff",
            "dcm"
        ],

        key="analysis_upload"
    )


    if analysis_upload is None:

        st.info(
            "Upload a medical image to analyze it."
        )


    else:

        try:

            (
                analysis_source,
                analysis_metadata

            ) = load_medical_image(
                analysis_upload
            )


            analysis_image = preprocess_image(
                analysis_source
            )


            analysis_features = extract_features(
                analysis_image
            )


            analysis_parameters = predict_parameters(
                analysis_features
            )


            left, right = st.columns(
                [
                    1,
                    1.7
                ]
            )


            with left:

                with st.container(
                    border=True
                ):

                    st.subheader(
                        "Standardized Image"
                    )


                    st.image(
                        analysis_image,
                        use_container_width=True,
                        clamp=True
                    )


                    st.metric(
                        "Entropy",
                        f"{analysis_features['entropy']:.5f}"
                    )


                    st.metric(
                        "Mean Intensity",
                        f"{analysis_features['mean']:.3f}"
                    )


                    st.metric(
                        "Edge Density",
                        f"{analysis_features['edge_density']:.5f}"
                    )


            with right:

                feature_tab, parameter_tab = st.tabs(
                    [
                        "Image Characteristics",
                        "Generated Parameters"
                    ]
                )


                with feature_tab:

                    dataframe = pd.DataFrame(

                        {

                            "Characteristic":
                                FEATURE_COLUMNS,

                            "Value":
                                [

                                    analysis_features[
                                        name
                                    ]

                                    for name
                                    in FEATURE_COLUMNS
                                ]
                        }
                    )


                    st.dataframe(

                        dataframe,

                        hide_index=True,

                        use_container_width=True,

                        height=630
                    )


                with parameter_tab:

                    parameter_dataframe = pd.DataFrame(

                        {

                            "Parameter":
                                list(
                                    analysis_parameters.keys()
                                ),

                            "Generated Value":
                                list(
                                    analysis_parameters.values()
                                )
                        }
                    )


                    st.dataframe(

                        parameter_dataframe,

                        hide_index=True,

                        use_container_width=True
                    )


            with st.expander(
                "Source Metadata"
            ):

                st.json(
                    analysis_metadata
                )


        except Exception as error:

            st.error(
                f"Unable to analyze image: {error}"
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()


footer_left, footer_center, footer_right = st.columns(
    [
        2,
        2,
        1
    ]
)


with footer_left:

    st.caption(
        f"{PRODUCT_NAME} · Medical Image Security Platform"
    )


with footer_center:

    st.caption(
        f"Cipher Engine: {CIPHER_VERSION}"
    )


with footer_right:

    st.caption(
        "Secure Workspace"
    )