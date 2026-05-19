from pathlib import Path
from setuptools import find_packages, setup

ROOT = Path(__file__).parent
long_description = (ROOT / "README.md").read_text(encoding="utf-8")
requirements = [
    line.strip()
    for line in (ROOT / "requirements.txt").read_text().splitlines()
    if line.strip() and not line.startswith("#")
]

setup(
    name="ssl-graph-anomaly",
    version="1.0.0",
    description=(
        "Self-Supervised Graph Neural Networks for Network Intrusion Detection "
        "with Conformal Safety Certification."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Roger Nick Anaedevha, Alexander Gennadievich Trofimov, Yuri Vladimirovich Borodachev",
    author_email="roger@robustidps.ai",
    url="https://github.com/rogerpanel/CV/tree/main/ssl_graph_anomaly",
    license="MIT",
    python_requires=">=3.10",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=requirements,
    extras_require={
        "full": ["torch-geometric>=2.5.0"],
        "dev": ["pytest>=8.2.0", "black>=24.4.0", "ruff>=0.4.0", "mypy>=1.10.0"],
    },
    entry_points={
        "console_scripts": [
            "sslga-train=ssl_graph_anomaly.training.pretrain:cli",
            "sslga-calibrate=ssl_graph_anomaly.training.calibrate:cli",
            "sslga-evaluate=ssl_graph_anomaly.evaluation.metrics:cli",
            "sslga-serve=ssl_graph_anomaly.inference.serve:cli",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security",
    ],
)
