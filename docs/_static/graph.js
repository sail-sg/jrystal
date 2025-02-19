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
              "v_local": "http://localhost:8000/index.html#module-jrystal.pseudopotential.local",
              "param_coeff": "http://localhost:8000/api/api.html#jrystal._src.pw.coeff"
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
