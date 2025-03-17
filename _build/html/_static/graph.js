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
              "e_kin": "api/energy.html#jrystal.energy.kinetic",
              "e_total": "api/energy.html#jrystal.energy.total_energy",
              "e_har": "api/energy.html#jrystal.energy.hartree",
              "e_ext": "api/energy.html#jrystal.energy.external",
              "e_xc": "api/energy.html#jrystal.energy.xc_lda",
              "e_ewald": "api/_src/ewald.html#jrystal._src.ewald.ewald_coulomb_repulsion",
              "E_free": "api/energy.html#jrystal.energy.free_energy",
              "Entropy": "api/entropy.html#jrystal.entropy.entropy",
              
              // Grid related
              "grid_size": "api/grid.html#jrystal.grid.grid_size",
              "gpts": "api/grid.html#jrystal.grid.g_vectors",
              "kpts": "api/grid.html#jrystal.grid.k_vectors",
              "freq_mask": "api/grid.html#jrystal.grid.frequency_mask",
              
              // Wave functions and coefficients
              "coeff": "api/pw.html#jrystal.pw.coeff",
              "param_coeff": "api/pw.html#jrystal.pw.param_init",
              "wave_r": "api/pw.html#jrystal.pw.wave_grid",
              "dens_r": "api/pw.html#jrystal.pw.density_grid",
              "dens_g": "api/pw.html#jrystal.pw.density_grid_reciprocal",
              
              // Occupation
              "occupation": "api/occupation.html#jrystal.occupation.occupation_numbers",
              "param_occ": "api/occupation.html#jrystal.occupation.param_occupation",
              
              // Potentials
              "v_local": "api/pseudopotential.html#jrystal.pseudopotential.local.local_potential",
              "v_nonlocal": "api/pseudopotential.html#jrystal.pseudopotential.beta.nonlocal_potential",
              "v_har": "api/potential.html#jrystal.potential.hartree",
              "v_xc": "api/potential.html#jrystal.potential.xc_lda",
              "v_ext": "api/potential.html#jrystal.potential.external",
              
              // Hamiltonian
              "hamil_matrix": "api/hamiltonian.html#jrystal.hamiltonian.hamiltonian_matrix",
              "hamil_matrix_diag": "api/hamiltonian.html#jrystal.hamiltonian.hamiltonian_matrix_trace",
              "band_structure": "api/energy.html#jrystal.energy.band_energy",
              
              // Crystal properties
              "cellvec": "api/grid.html#jrystal.grid.cell_vectors",
              "position": "api/grid.html#jrystal.grid.atomic_positions",
              "charge": "api/grid.html#jrystal.grid.atomic_charges",
              "spin": "api/grid.html#jrystal.grid.spin_configuration",
              "vol": "api/_src/utils.html#jrystal._src.utils.volume",
              "total_charge": "api/grid.html#jrystal.grid.total_charge",
              
              // Parameters
              "cutoff": "api/grid.html#jrystal.grid.cutoff_energy",
              "temperature": "api/entropy.html#jrystal.entropy.temperature",
              "ewald_eta": "api/ewald.html#jrystal.ewald.ewald_eta",
              "ewald_cutoff": "api/ewald.html#jrystal.ewald.ewald_cutoff",
            };
            // Lookup the URL from our lookup table; if not found, use a fallback.
            var docLinkURL = docLinkLookup[clickedLabel] ||
                             "http://localhost:8000/index.html#" + encodeURIComponent(clickedLabel);

            // Compute the position for the doc link
            var bbox = this.getBBox();
            
            // Find the text element in the node
            var nodeText = d3.select(this).select("text");
            var textX = parseFloat(nodeText.attr("x") || 0);
            var textY = parseFloat(nodeText.attr("y") || 0);
            
            // Calculate position for button below the node
            var buttonX = textX; // Center the button on the node
            var buttonY = textY + bbox.height + 5; // Position below the node using box height + small padding
            
            // Add the doc link directly to the node's group
            var link = d3.select(this)
              .append("svg:a")
              .attr("class", "doc-link")
              .attr("xlink:href", docLinkURL)
              .attr("target", "_blank");
            
            // Add background rectangle for button
            var buttonText = "go to doc";
            var padding = 6;
            var fontSize = 11;
            var buttonWidth = buttonText.length * fontSize * 0.6 + 2 * padding;
            var buttonHeight = fontSize + 2 * padding;
            
            link.append("rect")
              .attr("x", buttonX - buttonWidth/2) // Center the button horizontally
              .attr("y", buttonY)
              .attr("width", buttonWidth)
              .attr("height", buttonHeight)
              .style("fill", "#ffffff")        // White background
              .style("stroke", "#000000")      // Black border
              .style("stroke-width", "1px")
              .style("rx", "3")               // Rounded corners
              .style("ry", "3");
            
            // Add the text
            link.append("svg:text")
              .text(buttonText)
              .attr("x", buttonX) // Center the text
              .attr("y", buttonY + buttonHeight/2)
              .attr("dy", "0.3em")
              .attr("text-anchor", "middle") // Center text horizontally
              .style("font-size", fontSize + "px")
              .style("fill", "#0066cc");       // Blue text
          });
        });
    })
    .catch(function(error) {
      console.error("Error fetching DOT file:", error);
    });
});
