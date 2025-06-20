#[Written by Emil Skovgård (s235282) and Victor Clausen (s232604)]

from pathlib import Path
from typing import Dict

def modify_params(
    template_path: Path,
    param_changes: Dict[str, float],
    suffix: str = "_mod"
) -> str:
    """
    Reads an OceanWave3D .inp file, updates supported parameters, and writes
    a new file with `<original_filename><suffix>` in the same directory.

    Supported parameters:
      - `nsteps`: number of time steps (first value on line 5, index 4)
      - `gravity`: gravitational acceleration constant (first value on line 6, index 5)

    Args:
        template_path: Path to the original .inp file.
        param_changes: Dict where keys are parameter names (`nsteps` or `gravity`) and values are new floats.
        suffix: suffix to append to the original filename (default: `_mod`).

    Returns:
        A success message detailing the new file path and changes made.

    Raises:
        ValueError: If a requested parameter is unsupported or if lines cannot be found.
    """
    template_path = Path(template_path)
    # Add the _mod so that we can clearly see which file is edited. Important that it is "_mod"
    output_path = template_path.with_name(
        f"{template_path.name}{suffix}"
    )

    with template_path.open('r') as f:
        lines = f.readlines()

    modified_lines = list(lines)
    applied_changes = []

    # hardcoding the the line indexes for nsteps and gravity.
    #Could possibly have been dynamic, which cool be cool if we wished to do this for more conditions
    param_line_map = {
        'nsteps': 4,   # line index 4 (5th line)
        'gravity': 5,   # line index 5 (6th line)
    }

    for param_key, new_value in param_changes.items():
        key = param_key.lower()
        if key not in param_line_map:
            raise ValueError(f"Unsupported parameter `{param_key}`; supported: {', '.join(param_line_map)}.")

        line_idx = param_line_map[key]
        if len(modified_lines) <= line_idx:
            raise ValueError(f"Input file is too short - cannot find line for `{param_key}`")

        line = modified_lines[line_idx]
        values = line.split()
        if not values:
            raise ValueError(f"Line for `{param_key}` appears to be empty")

        old_value = values[0]
        # For gravity allow float; for nsteps cast to int
        if key == 'nsteps':
            values[0] = str(int(new_value))
        else:
            values[0] = str(new_value)

        # Preserve comment if present
        #Here because the agent should never change anything after "<-" in the .inp files
        comment_start = line.find('<-')
        data_part = ' '.join(values)
        if comment_start != -1:
            comment_part = line[comment_start:]
            modified_lines[line_idx] = f"{data_part:<40}{comment_part}"
        else:
            modified_lines[line_idx] = data_part + '\n'

        applied_changes.append(f"- Set `{param_key}` to `{values[0]}` (was `{old_value}`)")

    # Print the output path alongside the succes message for the user to see
    with output_path.open('w') as f:
        f.writelines(modified_lines)
    return (
        f"Successfully wrote modified file → {output_path!r}\n"
        + "Applied changes:\n"
        + "\n".join(applied_changes)
    )
