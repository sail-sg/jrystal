document.addEventListener("DOMContentLoaded", function() {

  fetch("/_static/graph.dot?" + new Date().getTime())
    .then(function(response) {
      if (!response.ok) {
        throw new Error("Network response was not ok: " + response.statusText);
      }
      return response.text();
    })
    .then(function(dot) {
      d3.select("#graph").graphviz().renderDot(dot)
        .on("end", function() {
          // Attach click events to every node.
          d3.selectAll("g.node").on("click", function(event) {
            // Retrieve the node's label from its <title> element.
            var clickedLabel = d3.select(this).select("title").text().trim();

            // Toggle: if the node is already active, clear highlighting and remove any doc links.
            if (d3.select(this).classed("active")) {
              d3.selectAll("g.node").classed("active", false).classed("dim", false);
              d3.selectAll("g.edge").classed("highlight", false).classed("dim", false);
              d3.selectAll("svg a.doc-link").remove();
              return;
            }

            // Clear any previous state.
            d3.selectAll("g.node").classed("active", false).classed("dim", false);
            d3.selectAll("g.edge").classed("highlight", false).classed("dim", false);
            d3.selectAll("svg a.doc-link").remove();

            // --- Compute the active chain (ancestors) ---
            // We'll use a recursive function to collect all ancestor labels.
            var activeNodes = [];
            function addAncestors(label) {
              if (activeNodes.indexOf(label) !== -1) return;
              activeNodes.push(label);
              // For each edge, if this label is the target, add its source.
              d3.selectAll("g.edge").each(function() {
                var edgeTitle = d3.select(this).select("title").text().trim();
                // Expecting edge title format "Source -> Target"
                var parts = edgeTitle.split("->");
                if (parts.length === 2) {
                  var source = parts[0].trim();
                  var target = parts[1].trim();
                  if (target === label) {
                    addAncestors(source);
                  }
                }
              });
            }
            addAncestors(clickedLabel);

            // --- Mark nodes based on active chain ---
            d3.selectAll("g.node").each(function() {
              var nodeLabel = d3.select(this).select("title").text().trim();
              if (activeNodes.indexOf(nodeLabel) !== -1) {
                d3.select(this).classed("active", true);
              } else {
                d3.select(this).classed("dim", true);
              }
            });

            // --- Mark edges: highlight if both endpoints are active, else dim ---
            d3.selectAll("g.edge").each(function() {
              var edgeTitle = d3.select(this).select("title").text().trim();
              var parts = edgeTitle.split("->");
              if (parts.length === 2) {
                var source = parts[0].trim();
                var target = parts[1].trim();
                if (activeNodes.indexOf(source) !== -1 && activeNodes.indexOf(target) !== -1) {
                  d3.select(this).classed("highlight", true);
                } else {
                  d3.select(this).classed("dim", true);
                }
              } else {
                d3.select(this).classed("dim", true);
              }
            });

            // --- Create the doc link ---
            // Define a lookup table for doc links.
            var docLinkLookup = {
              // Core modules - Energy related
              "e_kin": "api/core/kinetic.html#jrystal._src.kinetic.kinetic_energy",
              "e_total": "api/core/energy.html#jrystal._src.energy.total_energy",
              "e_har": "api/core/energy.html#jrystal._src.energy.hartree_energy",
              "e_ext": "api/core/energy.html#jrystal._src.energy.external_energy",
              "e_xc": "api/core/energy.html#jrystal._src.energy.exchange_correlation_energy",
              "e_ewald": "api/core/ewald.html#jrystal._src.ewald.ewald_energy",
              "E_free": "api/core/energy.html#jrystal._src.energy.free_energy",
              "Entropy": "api/core/entropy.html#jrystal._src.entropy.entropy",
              
              // Grid related
              "grid_size": "api/core/grid.html#jrystal._src.grid.grid_size",
              "gpts": "api/core/grid.html#jrystal._src.grid.g_vectors",
              "kpts": "api/core/grid.html#jrystal._src.grid.k_points",
              "freq_mask": "api/core/grid.html#jrystal._src.grid.frequency_mask",
              
              // Wave functions and coefficients
              "coeff": "api/core/pw.html#jrystal._src.pw.plane_wave_coefficients",
              "param_coeff": "api/core/pw.html#jrystal._src.pw.param_coefficients",
              "wave_r": "api/core/pw.html#jrystal._src.pw.wave_grid",
              "dens_r": "api/core/pw.html#jrystal._src.pw.density_grid",
              "dens_g": "api/core/pw.html#jrystal._src.pw.density_grid_reciprocal",
              
              // Occupation
              "occupation": "api/core/occupation.html#jrystal._src.occupation.occupation_numbers",
              "param_occ": "api/core/occupation.html#jrystal._src.occupation.param_occupation",
              
              // Potentials
              "v_local": "api/pseudopotential.html#jrystal.pseudopotential.local.local_potential",
              "v_nonlocal": "api/pseudopotential.html#jrystal.pseudopotential.beta.nonlocal_potential",
              "v_har": "api/core/potential.html#jrystal._src.potential.hartree_potential",
              "v_xc": "api/core/potential.html#jrystal._src.potential.exchange_correlation_potential",
              "v_ext": "api/core/potential.html#jrystal._src.potential.external_potential",
              
              // Hamiltonian
              "hamil_matrix": "api/core/hamiltonian.html#jrystal._src.hamiltonian.hamiltonian_matrix",
              "hamil_matrix_diag": "api/core/hamiltonian.html#jrystal._src.hamiltonian.hamiltonian_diagonal",
              "band_structure": "api/core/band.html#jrystal._src.band.band_structure",
              
              // Crystal properties
              "cellvec": "api/core/grid.html#jrystal._src.grid.cell_vectors",
              "position": "api/core/grid.html#jrystal._src.grid.atomic_positions",
              "charge": "api/core/grid.html#jrystal._src.grid.atomic_charges",
              "spin": "api/core/grid.html#jrystal._src.grid.spin_configuration",
              "vol": "api/core/grid.html#jrystal._src.grid.cell_volume",
              "total_charge": "api/core/grid.html#jrystal._src.grid.total_charge",
              
              // Parameters
              "cutoff": "api/core/grid.html#jrystal._src.grid.cutoff_energy",
              "temperature": "api/core/entropy.html#jrystal._src.entropy.temperature",
              "ewald_eta": "api/core/ewald.html#jrystal._src.ewald.ewald_eta",
              "ewald_cutoff": "api/core/ewald.html#jrystal._src.ewald.ewald_cutoff",
              
              // Band structure related
              "band_structure": "api/core/band.html#jrystal._src.band.band_structure",
              
              // Braket operations
              "braket": "api/core/braket.html#jrystal._src.braket",
              
              // Constants
              "const": "api/core/const.html#jrystal._src.const",
              
              // Crystal structure
              "crystal": "api/core/crystal.html#jrystal._src.crystal"
            };
            // Lookup the URL from our lookup table; if not found, use a fallback.
            var docLinkURL = docLinkLookup[clickedLabel] ||
                             "http://localhost:8000/index.html#" + encodeURIComponent(clickedLabel);

            // Compute the absolute position of the clicked node's right edge
            // with a constant offset in pixels.
            var bbox = this.getBBox();
            var ctm = this.getCTM();
            var scale = Math.sqrt(ctm.a * ctm.a + ctm.b * ctm.b);
            var desiredOffset = 5; // constant 5-pixel gap in screen space
            var offsetInLocal = desiredOffset / scale;
            var pt = this.ownerSVGElement.createSVGPoint();
            pt.x = bbox.x + bbox.width + offsetInLocal;
            pt.y = bbox.y + bbox.height / 2;
            var globalPt = pt.matrixTransform(ctm);

            // Append the doc link to the top-level SVG container.
            d3.select(this.ownerSVGElement)
              .append("svg:a")
              .attr("class", "doc-link")
              .attr("xlink:href", docLinkURL)
              .attr("target", "_blank")
              .append("svg:text")
              .text("doc")
              .attr("x", globalPt.x)
              .attr("y", globalPt.y);
          });
        });
    })
    .catch(function(error) {
      console.error("Error fetching DOT file:", error);
    });
});
