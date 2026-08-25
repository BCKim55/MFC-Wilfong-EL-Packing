from mfc.case_validator import CaseValidator


def _check_periodic_cloud(**overrides):
    params = {
        "ib": "F",
        "num_ibs": 0,
        "num_particle_clouds": 1,
        "particle_cloud(1)%num_particles": 8,
        "particle_cloud(1)%packing_method": 1,
        "particle_cloud(1)%periodic": 1,
        "particle_cloud(1)%length_x": 1.0,
        "particle_cloud(1)%length_y": 1.0,
        "p": 0,
    }
    params.update(overrides)
    validator = CaseValidator(params)
    validator.check_ibm()
    return validator.errors


def test_periodic_random_box_is_valid():
    assert not _check_periodic_cloud()


def test_periodic_rejects_non_box_packing():
    errors = _check_periodic_cloud(**{"particle_cloud(1)%packing_method": 3})
    assert any("only supported for random box packing" in error for error in errors)


def test_periodic_3d_requires_positive_z_length():
    errors = _check_periodic_cloud(**{"p": 31, "particle_cloud(1)%length_z": 0.0})
    assert any("requires positive box lengths" in error for error in errors)
