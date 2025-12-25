"""Unit tests for csiro_biomass.utils.io module."""

import pandas as pd
import pytest
import yaml

from csiro_biomass.utils.io import read_csv, read_yaml, write_csv


class TestReadYaml:
    """Tests for read_yaml function."""

    def test_read_yaml_valid_file(self, tmp_path, sample_config_dict):
        """Test reading a valid YAML file."""
        yaml_file = tmp_path / "config.yaml"
        with open(yaml_file, "w", encoding="utf-8") as f:
            yaml.dump(sample_config_dict, f)

        result = read_yaml(yaml_file)

        assert result == sample_config_dict
        assert result["general"]["seed"] == 42
        assert result["trainer"]["batch_size"] == 8

    def test_read_yaml_empty_file(self, tmp_path):
        """Test reading an empty YAML file."""
        yaml_file = tmp_path / "empty.yaml"
        yaml_file.write_text("", encoding="utf-8")

        result = read_yaml(yaml_file)

        assert result is None

    def test_read_yaml_nested_structure(self, tmp_path):
        """Test reading YAML with nested structure."""
        yaml_content = {
            "level1": {"level2": {"level3": {"value": "deep"}}, "simple": "value"}
        }
        yaml_file = tmp_path / "nested.yaml"
        with open(yaml_file, "w", encoding="utf-8") as f:
            yaml.dump(yaml_content, f)

        result = read_yaml(yaml_file)

        assert result["level1"]["level2"]["level3"]["value"] == "deep"
        assert result["level1"]["simple"] == "value"

    def test_read_yaml_file_not_found(self):
        """Test reading a non-existent YAML file."""
        with pytest.raises(FileNotFoundError):
            read_yaml("non_existent_file.yaml")

    def test_read_yaml_invalid_yaml(self, tmp_path):
        """Test reading an invalid YAML file."""
        yaml_file = tmp_path / "invalid.yaml"
        yaml_file.write_text("invalid: yaml: content: ::::", encoding="utf-8")

        with pytest.raises(yaml.YAMLError):
            read_yaml(yaml_file)

    def test_read_yaml_with_lists(self, tmp_path):
        """Test reading YAML with lists."""
        yaml_content = {"items": [1, 2, 3], "names": ["a", "b", "c"]}
        yaml_file = tmp_path / "lists.yaml"
        with open(yaml_file, "w", encoding="utf-8") as f:
            yaml.dump(yaml_content, f)

        result = read_yaml(yaml_file)

        assert result["items"] == [1, 2, 3]
        assert result["names"] == ["a", "b", "c"]


class TestReadCsv:
    """Tests for read_csv function."""

    def test_read_csv_valid_file(self, tmp_path):
        """Test reading a valid CSV file."""
        csv_file = tmp_path / "data.csv"
        df = pd.DataFrame(
            {"image_path": ["img1.jpg", "img2.jpg"], "target_name": ["A", "B"], "target": [1.5, 2.5]}
        )
        df.to_csv(csv_file, index=False)

        result = read_csv(csv_file)

        pd.testing.assert_frame_equal(result, df)

    def test_read_csv_empty_file(self, tmp_path):
        """Test reading an empty CSV file."""
        csv_file = tmp_path / "empty.csv"
        csv_file.write_text("", encoding="utf-8")

        result = read_csv(csv_file)

        assert result.empty

    def test_read_csv_with_numeric_data(self, tmp_path):
        """Test reading CSV with numeric data."""
        csv_file = tmp_path / "numeric.csv"
        df = pd.DataFrame(
            {
                "col1": [1, 2, 3],
                "col2": [4.5, 5.5, 6.5],
                "col3": ["a", "b", "c"],
            }
        )
        df.to_csv(csv_file, index=False)

        result = read_csv(csv_file)

        pd.testing.assert_frame_equal(result, df)
        assert result["col1"].dtype == "int64"
        assert result["col2"].dtype == "float64"

    def test_read_csv_file_not_found(self):
        """Test reading a non-existent CSV file."""
        with pytest.raises(FileNotFoundError):
            read_csv("non_existent_file.csv")

    def test_read_csv_large_file(self, tmp_path):
        """Test reading a large CSV file."""
        csv_file = tmp_path / "large.csv"
        df = pd.DataFrame({"col1": range(1000), "col2": range(1000, 2000)})
        df.to_csv(csv_file, index=False)

        result = read_csv(csv_file)

        assert len(result) == 1000
        pd.testing.assert_frame_equal(result, df)

    def test_read_csv_with_missing_values(self, tmp_path):
        """Test reading CSV with missing values."""
        csv_file = tmp_path / "missing.csv"
        df = pd.DataFrame({"col1": [1, None, 3], "col2": ["a", "b", None]})
        df.to_csv(csv_file, index=False)

        result = read_csv(csv_file)

        assert result.isna().sum().sum() == 2


class TestWriteCsv:
    """Tests for write_csv function."""

    def test_write_csv_valid_dataframe(self, tmp_path):
        """Test writing a valid DataFrame to CSV."""
        csv_file = tmp_path / "output.csv"
        df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})

        write_csv(csv_file, df)

        assert csv_file.exists()
        result = pd.read_csv(csv_file)
        pd.testing.assert_frame_equal(result, df)

    def test_write_csv_empty_dataframe(self, tmp_path):
        """Test writing an empty DataFrame to CSV."""
        csv_file = tmp_path / "empty_output.csv"
        df = pd.DataFrame()

        write_csv(csv_file, df)

        assert csv_file.exists()
        result = pd.read_csv(csv_file)
        assert result.empty

    def test_write_csv_overwrites_existing(self, tmp_path):
        """Test that write_csv overwrites existing files."""
        csv_file = tmp_path / "overwrite.csv"
        df1 = pd.DataFrame({"col1": [1, 2]})
        df2 = pd.DataFrame({"col1": [3, 4, 5]})

        write_csv(csv_file, df1)
        write_csv(csv_file, df2)

        result = pd.read_csv(csv_file)
        pd.testing.assert_frame_equal(result, df2)
        assert len(result) == 3

    def test_write_csv_with_special_characters(self, tmp_path):
        """Test writing CSV with special characters."""
        csv_file = tmp_path / "special.csv"
        df = pd.DataFrame({"col1": ["a,b", "c\"d", "e\nf"]})

        write_csv(csv_file, df)

        result = pd.read_csv(csv_file)
        pd.testing.assert_frame_equal(result, df)

    def test_write_csv_preserves_types(self, tmp_path):
        """Test that write_csv preserves data types on round-trip."""
        csv_file = tmp_path / "types.csv"
        df = pd.DataFrame(
            {
                "int_col": [1, 2, 3],
                "float_col": [1.1, 2.2, 3.3],
                "str_col": ["a", "b", "c"],
            }
        )

        write_csv(csv_file, df)
        result = pd.read_csv(csv_file)

        assert result["int_col"].dtype == "int64"
        assert result["float_col"].dtype == "float64"
        assert result["str_col"].dtype == "object"

    def test_write_csv_with_index(self, tmp_path):
        """Test write_csv doesn't include index by default."""
        csv_file = tmp_path / "no_index.csv"
        df = pd.DataFrame({"col1": [1, 2, 3]})

        write_csv(csv_file, df)

        with open(csv_file, encoding="utf-8") as f:
            first_line = f.readline()
            assert first_line.strip() == "col1"


class TestIntegrationIo:
    """Integration tests for IO operations."""

    def test_read_write_csv_roundtrip(self, tmp_path):
        """Test reading and writing CSV in a roundtrip."""
        csv_file = tmp_path / "roundtrip.csv"
        original_df = pd.DataFrame(
            {
                "image_path": ["img1.jpg", "img2.jpg", "img3.jpg"],
                "target_name": ["A", "B", "C"],
                "target": [1.5, 2.5, 3.5],
            }
        )

        write_csv(csv_file, original_df)
        result_df = read_csv(csv_file)

        pd.testing.assert_frame_equal(result_df, original_df)

    def test_yaml_csv_combined_workflow(self, tmp_path):
        """Test a workflow combining YAML and CSV operations."""
        yaml_file = tmp_path / "config.yaml"
        config = {"dataset": {"train": str(tmp_path / "train.csv")}}
        with open(yaml_file, "w", encoding="utf-8") as f:
            yaml.dump(config, f)

        csv_file = tmp_path / "train.csv"
        df = pd.DataFrame({"col1": [1, 2, 3]})
        write_csv(csv_file, df)

        config_result = read_yaml(yaml_file)
        df_result = read_csv(config_result["dataset"]["train"])

        assert config_result["dataset"]["train"] == str(csv_file)
        pd.testing.assert_frame_equal(df_result, df)

    def test_multiple_csv_files(self, tmp_path):
        """Test handling multiple CSV files."""
        files = []
        for i in range(3):
            csv_file = tmp_path / f"data_{i}.csv"
            df = pd.DataFrame({"value": [i * 10, i * 10 + 1, i * 10 + 2]})
            write_csv(csv_file, df)
            files.append(csv_file)

        dfs = [read_csv(f) for f in files]

        assert len(dfs) == 3
        assert dfs[0]["value"].tolist() == [0, 1, 2]
        assert dfs[1]["value"].tolist() == [10, 11, 12]
        assert dfs[2]["value"].tolist() == [20, 21, 22]
