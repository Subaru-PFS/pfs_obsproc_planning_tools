(function(cell) {
    const prefix = cell.getField().startsWith("sky") ? "sky" : "std";
    const rowData = cell.getRow().getData();
    const trigger = document.createElement("span");
    trigger.className = "validation-tooltip-trigger";
    if (rowData[`_${prefix}_total_warning`]) {
        trigger.classList.add(
            "validation-tooltip-warning",
            `validation-tooltip-warning-${prefix}`,
        );
    }
    trigger.textContent = cell.getValue();

    let tooltip = null;
    const moveTooltip = (event) => {
        if (!tooltip) return;
        tooltip.style.left = `${event.clientX + 14}px`;
        tooltip.style.top = `${event.clientY + 14}px`;
    };
    trigger.addEventListener("mouseenter", (event) => {
        tooltip = document.createElement("div");
        tooltip.className = "validation-row-tooltip";
        tooltip.innerHTML = rowData[`_${prefix}_details`];
        document.body.appendChild(tooltip);
        moveTooltip(event);
    });
    trigger.addEventListener("mousemove", moveTooltip);
    trigger.addEventListener("mouseleave", () => {
        tooltip?.remove();
        tooltip = null;
    });
    return trigger;
})