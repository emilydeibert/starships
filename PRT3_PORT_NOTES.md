# petitRADTRANS 3 port notes

Base branch: upstream cleanup
Development branch: prt3-retrieval

## Already partially compatible

- starships/retrieval.py imports Guillot profiles from
  petitRADTRANS.physics with a pRT2 fallback.
- starships/retrieval_utils.py does the same.

## Known pRT2 dependencies

### starships/petitradtrans_utils.py

- pRT2 Radtrans import
- petitRADTRANS.nat_cst
- poor_mans_nonequ_chem.interpol_abundances
- pRT2 Radtrans constructor arguments
- atmosphere.setup_opa_structure(pressures)
- atmosphere.calc_flux(...)
- atmosphere.flux
- atmosphere.freq

## Initial port scope

1. Replace imports.
2. Port Radtrans initialization.
3. Port line-species identifiers.
4. Port equilibrium-chemistry interface.
5. Port emission-spectrum calculation.
6. Preserve the wavelength/model return interface used by retrieval.py.
7. Add a forward-model regression test before attempting a sampler run.
