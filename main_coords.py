import csv
import math
import numpy as np
import scipy.constants as con
from scipy.optimize import minimize

# coordinates of M13
c_ra = 250.423475
c_dec = 36.46131944

# the limiting search radius for the Arecibo beam was taken as ~29.88 deg
# the limiting search radius for a 25-m paraboloid beam was taken as ~30.198 deg
# the Arecibo sample is contained entirely within the 25-m paraboloid sample.
#
# the following Gaia archive ADQL query fetches the required parameters for Gaia DR2
# sources within 30.198 deg of the J2015.5 coordinates of M13:
#
# SELECT source_id, dist.r_est, dist.r_lo, dist.r_hi,
# src.ra, src.ra_error, src.dec, src.dec_error,
# src.pmra, src.pmra_error, src.pmdec, src.pmdec_error,
# src.phot_g_mean_mag, src.bp_rp, src.teff_val, src.a_g_val
# FROM external.gaiadr2_geometric_distance AS dist
# JOIN (SELECT * FROM gaiadr2.gaia_source
# WHERE CONTAINS(
# POINT('ICRS',gaiadr2.gaia_source.ra,gaiadr2.gaia_source.dec),
# CIRCLE( 'ICRS', COORD1(EPOCH_PROP_POS(250.42347500000002,36.46131944,.0813,-3.1800,-2.5600,-244.4900,2000,2015.5)),
# COORD2(EPOCH_PROP_POS(250.42347500000002,36.46131944,.0813,-3.1800,-2.5600,-244.4900,2000,2015.5)), 30.198) )=1)
# AS subquery USING (source_id)
# JOIN gaiadr2.gaia_source AS src USING (source_id)
#
# the provided "test.csv" file under possible_targets is the first five rows of the outputted table,
# to indicate the required formatting

possible_targets = "test.csv"

# fun[0] = ra
# fun[1] = dec
# fun[2] = pmra
# fun[3] = pmdec
# fun[4] = r


def gcd(fun):
    # great circle angle in radians between ra, dec and c_ra, c_dec at time t
    delta_theta = np.arccos(np.sin(fun[1] + fun[3] * (fun[4] / con.c)) * np.sin(np.radians(c_dec)) +
                            np.cos(fun[1] + fun[3] * (fun[4] / con.c)) * np.cos(np.radians(c_dec)) *
                            np.cos(fun[0] + fun[2] * (fun[4] / con.c) - np.radians(c_ra)))
    return delta_theta


def neg_gcd(fun):
    # negative great circle angle in radians between ra, dec and c_ra, c_dec at time t
    delta_theta = (np.arccos(np.sin(fun[1] + fun[3] * (fun[4] / con.c)) * np.sin(np.radians(c_dec)) +
                   np.cos(fun[1] + fun[3] * (fun[4] / con.c)) * np.cos(np.radians(c_dec)) *
                   np.cos(fun[0] + fun[2] * (fun[4] / con.c) - np.radians(c_ra)))) * -1
    return delta_theta


source_ids = []
r_ests = []
r_los = []
r_his = []
ras = []
ra_errors = []
decs = []
dec_errors = []
pmras = []
pmra_errors = []
pmdecs = []
pmdec_errors = []
phot_g_mean_mags = []
bp_rps = []
teff_vals = []
a_g_vals = []
est_gcds = []

closest_ra = []
closest_dec = []
closest_pmra = []
closest_pmdec = []
closest_dist = []
closest_gcd = []
furthest_ra = []
furthest_dec = []
furthest_pmra = []
furthest_pmdec = []
furthest_dist = []
furthest_gcd = []

skipped_count = 0
with open(possible_targets, "r") as csvfile:
    datareader = csv.reader(csvfile)
    next(datareader)
    processed_count = 1
    for row in datareader:
        try:
            # parse row values
            source_id = row[0]
            r_est = float(row[1])
            r_lo = float(row[2])
            r_hi = float(row[3])
            ra = float(row[4])
            ra_error = float(row[5])
            dec = float(row[6])
            dec_error = float(row[7])
            pmra = float(row[8])
            pmra_error = float(row[9])
            pmdec = float(row[10])
            pmdec_error = float(row[11])
            phot_g_mean_mag = float(row[12])
            bp_rp = float(row[13])
            teff_val = float(row[14])
            a_g_val = float(row[15])

            # COORDINATES IN DEG, ERRORS IN MAS >> CONVERT TO RAD
            ra_min = math.radians(ra - (ra_error / 3.6e6))
            ra_max = math.radians(ra + (ra_error / 3.6e6))
            dec_min = math.radians(dec - (dec_error / 3.6e6))
            dec_max = math.radians(dec + (dec_error / 3.6e6))
            # PM IN MAS/YR, PM ERROR IN MAS/YR >> CONVERT TO RAD/S
            pmra_min = math.radians((pmra - pmra_error) / 3.6e6) / 3.154e7
            pmra_max = math.radians((pmra + pmra_error) / 3.6e6) / 3.154e7
            pmdec_min = math.radians((pmdec - pmdec_error) / 3.6e6) / 3.154e7
            pmdec_max = math.radians((pmdec + pmdec_error) / 3.6e6) / 3.154e7
            # DISTANCE IN PC >> CONVERT TO M
            r_est_min = r_lo * 3.086e+16
            r_est_max = r_hi * 3.086e+16

            ra_guess = math.radians(ra)
            dec_guess = math.radians(dec)
            pmra_guess = math.radians((pmra) / 3.6e6) / 3.154e7
            pmdec_guess = math.radians((pmdec) / 3.6e6) / 3.154e7
            r_guess = r_est * 3.086e+16
            est_gcd = np.arccos(np.sin(dec_guess + pmdec_guess * (r_guess / con.c)) * np.sin(np.radians(c_dec))
                                + np.cos(dec_guess + pmdec_guess * (r_guess / con.c)) * np.cos(np.radians(c_dec))
                                * np.cos(ra_guess + pmra_guess * (r_guess / con.c) - np.radians(c_ra)))

            start_guess = [ra_guess, dec_guess, pmra_guess, pmdec_guess, r_guess]

            my_bounds = ((ra_min, ra_max), (dec_min, dec_max), (pmra_min, pmra_max), (pmdec_min, pmdec_max),
                         (r_est_min, r_est_max))

            n_min = minimize(gcd, x0=start_guess, method='Nelder-Mead', bounds=my_bounds)
            n_max = minimize(neg_gcd, x0=start_guess, method='Nelder-Mead', bounds=my_bounds)

            source_ids.append(source_id)
            r_ests.append(r_est)
            r_los.append(r_lo)
            r_his.append(r_hi)
            ras.append(ra)
            ra_errors.append(ra_error)
            decs.append(dec)
            dec_errors.append(dec_error)
            pmras.append(pmra)
            pmra_errors.append(pmra_error)
            pmdecs.append(pmdec)
            pmdec_errors.append(pmdec_error)
            phot_g_mean_mags.append(phot_g_mean_mag)
            bp_rps.append(bp_rp)
            teff_vals.append(teff_val)
            a_g_vals.append(a_g_val)
            est_gcds.append(math.degrees(est_gcd))
            # convert all parameters back to original units
            closest_ra.append(math.degrees(n_min['x'][0]))
            closest_dec.append(math.degrees(n_min['x'][1]))
            closest_pmra.append(n_min['x'][2] * 3.154e7 * 3.6e6)
            closest_pmdec.append(n_min['x'][3] * 3.154e7 * 3.6e6)
            closest_dist.append(n_min['x'][4] * 3.24078e-17)
            closest_gcd.append(math.degrees(n_min["fun"]))
            furthest_ra.append(math.degrees(n_max['x'][0]))
            furthest_dec.append(math.degrees(n_max['x'][1]))
            furthest_pmra.append(n_max['x'][2] * 3.154e7 * 3.6e6)
            furthest_pmdec.append(n_max['x'][3] * 3.154e7 * 3.6e6)
            furthest_dist.append(n_max['x'][4] * 3.24078e-17)
            furthest_gcd.append(math.degrees(n_max["fun"]) * -1)
            print("Target {} completed.".format(processed_count))
            processed_count += 1
        except ValueError:
            skipped_count += 1
            pass

with open("meti_error_output.csv", "w") as csvfile:
    lines = []
    writer = csv.writer(csvfile)
    headers = ["source_id",
               "r_est_(pc)", "r_lo_(pc)", "r_hi_(pc)",
               "ra_(deg)", "ra_error_(mas)",
               "dec_(deg)", "dec_error_(mas)",
               "pmra_(mas/yr)", "pmra_error_(mas/yr)",
               "pmdec_(mas/yr)", "pmdec_error_(mas/yr)",
               "phot_g_mean_mag_(mag)", "bp_rp_(mag)", "teff_val_(K)", "a_g_val_(mag)",
               "est_gcd_(deg)",
               "closest_ra_(deg)", "closest_dec_(deg)",
               "closest_pmra_(mas/yr)", "closest_pmdec_(mas/yr)",
               "closest_dist_(pc)", "closest_gcd_(deg)",
               "furthest_ra_(deg)", "furthest_dec_(deg)",
               "furthest_pmra_(mas/yr)", "furthest_pmdec_(mas/yr)",
               "furthest_dist_(pc)", "furthest_gcd_(deg)"]

    writer.writerow(headers)
    data = list(zip(source_ids, r_ests, r_los, r_his, ras, ra_errors,
                    decs, dec_errors, pmras, pmra_errors, pmdecs, pmdec_errors,
                    phot_g_mean_mags, bp_rps, teff_vals, a_g_vals, est_gcds,
                    closest_ra, closest_dec, closest_pmra, closest_pmdec, closest_dist, closest_gcd,
                    furthest_ra, furthest_dec, furthest_pmra, furthest_pmdec, furthest_dist, furthest_gcd))
    for row in data:
        row = list(row)
        writer.writerow(row)
    print("All rows written to 'meti_error_output.csv'.")
    print("{} targets with no estimated distance skipped.".format(skipped_count))
