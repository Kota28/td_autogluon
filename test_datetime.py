from autogluon.features.generators import DatetimeFeatureGenerator


def test_datetime_feature_generator(generator_helper, data_helper):
    # Given
    input_data = data_helper.generate_multi_feature_full()

    generator_1 = DatetimeFeatureGenerator()

    expected_feature_metadata_in_full = {
        ('datetime', ()): ['datetime'],
        ('object', ('datetime_as_object',)): ['datetime_as_object'],
    }

    expected_feature_metadata_full_1 = {('int', ('datetime_as_int',)): [
        'datetime',
        'datetime.year',
        'datetime.month',
        'datetime.day',
        'datetime.dayofweek',
        'datetime.hour',
        'datetime_as_object',
        'datetime_as_object.year',
        'datetime_as_object.month',
        'datetime_as_object.day',
        'datetime_as_object.dayofweek',
        'datetime_as_object.hour'
    ]}


    expected_output_data_feat_datetime = [
        1533140820000000000,
        1301322000000000000,
        1524238620000000000,
        1524238620000000000,
        -5364662400000000000,
        7289654340000000000,
        1301322000000000000
    ]

    expected_output_data_feat_datetime_year = [
        2018,
        2011,
        2018,
        2018,
        1800,
        2200,
        2011
    ]

    expected_output_data_feat_datetime_hour = [
        16,
        14,
        15,
        15,
        0,
        23,
        14
    ]

    # When
    output_data_1 = generator_helper.fit_transform_assert(
        input_data=input_data,
        generator=generator_1,
        expected_feature_metadata_in_full=expected_feature_metadata_in_full,
        expected_feature_metadata_full=expected_feature_metadata_full_1,
    )

    assert list(output_data_1['datetime'].values) == list(output_data_1['datetime_as_object'].values)
    assert expected_output_data_feat_datetime == list(output_data_1['datetime'].values)
    assert expected_output_data_feat_datetime_year == list(output_data_1['datetime.year'].values)
    assert expected_output_data_feat_datetime_hour == list(output_data_1['datetime.hour'].values)