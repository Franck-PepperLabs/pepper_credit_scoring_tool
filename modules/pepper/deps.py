import pkg_resources
from typing import List, Dict, Tuple, Union


def get_dependency_tree(
    root_pkg_name: str,
    verbose: bool=False
) -> Dict[str, Dict]:
    """Returns the dependency tree of the root package, where each node is
    represented by a dictionary with keys "requirement", "version", and
    "dependencies". The value of "version" is a string representing the
    version of the package, the value of "dependencies" is a dictionary
    of its dependencies each represented in the same format.

    Raises
    ------
    ValueError
        If a circular dependency is detected.

    Parameters
    ----------
    root_pkg_name : str
        The name of the root package.
    verbose : bool, optional
        If True, prints detailed information about each package and its
        dependencies (default is False).

    Returns
    -------
    A dictionary representing the dependency tree of the root package.
    Each key is a package name and each value is a dictionary representing its
    version, dependencies and python version.
    """
    def _get_dependency_tree(
        pkg_name: str,
        ancestor_pkg_names: List[str] = []
    ) -> Dict[str, Tuple[str, Union[Dict, None]]]:
        # Dict[str, Tuple[str, Union[Dict[str, Tuple[str, str, Union[Dict, None]]]], str]]]:
        """Recursively builds the dependency tree of a package.

        Parameters
        ----------
        pkg_name : str
            The name of the package.
        ancestor_pkg_names : List[str]
            The names of ancestor packages, to check for circular dependencies.

        Returns
        -------
        A tuple representing the dependency tree of the package, where the
        first element is the version of the package, the second element is
        a dictionary representing its dependencies.
        """
        # Check for circular dependencies
        if pkg_name in ancestor_pkg_names:
            raise ValueError(
                "Circular dependency detected: "
                + " -> ".join(ancestor_pkg_names + [pkg_name])
            )
        
        # Get dependencies
        pkg = pkg_resources.get_distribution(pkg_name)
        version = pkg.version
        reqs = pkg.requires()
        if verbose:
            depth = len(ancestor_pkg_names)
            indent = "  " * depth
            print(f"{indent}{'/'.join(ancestor_pkg_names)}/{pkg}")
            [print(f"{indent}  {req}") for req in reqs]
        
        # Build the dependency tree recursively
        deps_tree = None
        if reqs:
            deps_tree = {}
            for req in reqs:
                child_pkg_name = req.project_name
                child = deps_tree[child_pkg_name] = {}
                child["requirement"] = str(req)
                child_vd = _get_dependency_tree(
                    child_pkg_name,
                    ancestor_pkg_names + [pkg_name]
                )
                child["version"], child["dependencies"] = child_vd

        #return version, deps_tree
        return version, deps_tree
    
    # Get the dependency tree
    root_version, root_dependencie = _get_dependency_tree(root_pkg_name)
    return {
        root_pkg_name: {
            "version": root_version,
            "dependencies": root_dependencie
        }
    }
