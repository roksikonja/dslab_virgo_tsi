## Data Science With Total Solar Irradiance from SOHO Spacecraft

Data Science Lab Project - VIRGO Total Solar Irradiance

Project by **Luka Kolar, Rok Šikonja and Lenart Treven**.

### Dataset

Dataset was downloaded from publicly available server: [link](ftp://ftp.pmodwrc.ch/pub/data/irradiance/virgo/).
Dataset description can be found [here](https://www.pmodwrc.ch/en/research-development/space/virgo-soho/).

### Usage

**Note**: Before use please add file `Virgo_Level1.txt` to folder `data`.

```
# modeling_1.py
python modeling_1.py --model_type="exp" --window=20  // exp. model
python modeling_1.py --model_type="exp_lin" --window=20  // exp. lin. model

// TODO: Lenart
python modeling_1.py --model_type="svr" --window=20 --param1 ... --paramK // SVM regressor
python modeling_1.py --model_type="cubic_spline" --window=20 --param1 ... --paramK // Spline model
python modeling_1.py --model_type="gpr" --window=20 --param1 ... --paramK // GP model
```