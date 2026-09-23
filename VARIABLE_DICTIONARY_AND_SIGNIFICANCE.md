# 📘 Variable Dictionary, Meanings & Physical Significance

This document provides a comprehensive dictionary of all raw meteorological parameters, engineered physics features, thermodynamic constants, model targets, and volumetric water output variables used in **Hydra: AI-Based Reservoir Evaporation & Water Level Prediction System**.

---

## 1. Raw Ingested Weather Variables

| Variable Name | Full Name | Unit | Physical Meaning | Technical & Hydrological Significance |
|---|---|---|---|---|
| `date` | Observation Date | `YYYY-MM-DD` | Timestamp of the daily meteorological observation. | Preserves temporal sequence integrity; used to engineer Day-of-Year (`doy`) features and track seasonal patterns. |
| `temp_max_c` | Maximum Temperature | `°C` | Highest air temperature recorded during a 24-hour period. | Determines upper saturation vapor pressure boundary $e^0(T_{max})$. Peak daily heat drives maximum evaporation potential. |
| `temp_min_c` | Minimum Temperature | `°C` | Lowest air temperature recorded during a 24-hour period. | Determines night-time cooling and lower saturation vapor pressure boundary $e^0(T_{min})$. |
| `temp_mean_c` | Mean Air Temperature | `°C` | Average temperature over 24 hours: $(T_{max} + T_{min}) / 2$. | Primary thermal indicator governing latent heat of vaporization ($\lambda \approx 2.45 \text{ MJ/kg}$) and average kinetic energy of water molecules. |
| `humidity_pct` | Relative Humidity | `%` | Ratio of actual vapor pressure to saturation vapor pressure at temperature $T$. | Governs atmospheric moisture saturation. Low humidity creates a steep Vapor Pressure Deficit ($VPD$), accelerating evaporation rate. |
| `wind_speed_10m_ms` | Wind Speed at 10m | `m/s` | Raw wind speed measured at standard weather station anemometer height ($10\text{ m}$). | Measures regional atmospheric circulation velocity. |
| `wind_speed_2m_ms` | Wind Speed at 2m ($u_2$) | `m/s` | Converted wind speed at $2\text{ m}$ height above ground/water surface. | **Crucial Aerodynamic Parameter**: Sweeps away the saturated boundary air layer immediately above the reservoir surface. Converted via logarithmic profile: $u_2 = 0.748 \cdot u_{10}$. |
| `solar_radiation_mj` | Shortwave Solar Radiation ($R_s$) | `MJ/m²/day` | Total shortwave solar energy reaching a horizontal surface per day. | **Primary Energy Driver ($r = 0.91$)**: Provides the thermal latent heat required to break liquid hydrogen bonds and evaporate water molecules. |
| `sunshine_hours` | Bright Sunshine Duration | `hours/day` | Duration of bright, unclouded sun exposure per day. | Direct indicator of cloud-free solar radiation coverage. |
| `pressure_hpa` | Atmospheric Surface Pressure | `hPa` | Mean barometric air pressure at ground elevation. | Dictates atmospheric density and psychrometric constant ($\gamma = 0.665 \cdot 10^{-3} \cdot P$). |
| `rainfall_mm` | Daily Precipitation | `mm/day` | Total depth of rainfall received during the day. | Water input variable. Rain days increase relative humidity to near 100% and reduce solar radiation, causing sharp drops in daily evaporation. |

---

## 2. Engineered Physics & Thermodynamic Variables

| Variable Name | Full Name | Unit | Formula / Derivation | Physical & Modeling Significance |
|---|---|---|---|---|
| `vpd_kpa` | Vapor Pressure Deficit ($VPD$) | `kPa` | $VPD = e_s - e_a$ | **Direct Drying Power Index**: Measures the atmosphere's capacity to absorb additional water vapor. High $VPD$ strongly drives evaporation. |
| `net_radiation_mj` | Net Surface Radiation ($R_n$) | `MJ/m²/day` | $R_n = (1 - \alpha) R_s - R_{nl}$ | Thermal energy remaining at the open water surface after accounting for albedo reflection ($\alpha \approx 0.08$) and net longwave thermal radiation loss ($R_{nl}$). |
| $e_s$ | Saturation Vapor Pressure | `kPa` | $\frac{e^0(T_{max}) + e^0(T_{min})}{2}$ | Maximum partial pressure water vapor can exert before condensation occurs at given temperatures. |
| $e_a$ | Actual Vapor Pressure | `kPa` | $e_s \cdot \left(\frac{RH}{100}\right)$ | Actual partial pressure exerted by water vapor molecules present in ambient air. |
| $\Delta$ | Slope of Saturation Curve | `kPa/°C` | $\frac{4098 \cdot e^0(T)}{(T + 237.3)^2}$ | Thermodynamic weighting factor in FAO-56 equation determining relative influence of radiation vs aerodynamic terms. |
| $\gamma$ | Psychrometric Constant | `kPa/°C` | $0.665 \cdot 10^{-3} \cdot P$ | Relates sensible heat loss to latent heat transfer based on atmospheric pressure. |

---

## 3. Target Variables & Model Outputs

| Variable Name | Full Name | Unit | Source / Model Origin | Physical Meaning & Utility |
|---|---|---|---|---|
| `penman_evaporation_mm` | Penman-Monteith Evaporation ($E_0$) | `mm/day` | Solved via FAO-56 Penman-Monteith equation | Pure thermodynamic physics benchmark evaporation depth. |
| `evaporation_pan_mm` | Class A Pan Evaporation ($E_{pan}$) | `mm/day` | $E_{pan} = E_0 / K_p$ (where $K_p = 0.75$) | **Target Variable ($Y$)**: Calibrated ground-truth evaporation depth representing Class A Evaporation Pan readings. |
| `predicted_evaporation_mm` | Model Forecasted Evaporation | `mm/day` | Generated by trained ML/DL models (MLP, SVR, GBDT) | Machine learning predicted daily evaporation depth for any target location or live 7-day forecast. |

---

## 4. Volumetric Water Loss & Societal Output Variables

| Variable Name | Full Name | Unit | Derivation | Real-World Decision Significance |
|---|---|---|---|---|
| `surface_area` | Reservoir Surface Area ($A$) | `km²` | Input parameter (e.g. Khadakwasla = $14.8\text{ km}^2$) | Spatial area over which open surface evaporation occurs. |
| `volume_loss_liters` | Volume Lost in Liters | `Liters/day` | $\text{Evaporation (mm)} \times A (\text{km}^2) \times 10^6$ | Converts depth loss into total daily volume of freshwater lost ($1\text{ mm}$ over $1\text{ km}^2 = 1,000,000\text{ Liters}$). |
| `volume_loss_mcm` | Volume Lost in MCM | `MCM/day` | $\text{Volume (Liters)} / 10^9$ | Standard irrigation engineering unit: **Million Cubic Meters** ($1\text{ MCM} = 10^9\text{ Liters}$). |
| `pune_supply_days_equiv` | City Supply Equivalence | `Days` | $\frac{\text{Volume Lost (Liters)}}{1.4 \times 10^9 \text{ MLD}}$ | Translates volumetric loss into equivalent days of municipal drinking water supply for Pune City ($1,400\text{ MLD}$ demand), aiding municipal water authorities. |
