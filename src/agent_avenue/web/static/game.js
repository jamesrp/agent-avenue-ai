(() => {
  const builder = document.querySelector("[data-offer-builder]");
  if (!(builder instanceof HTMLFormElement)) return;

  const cards = Array.from(builder.querySelectorAll("[data-hand-card]"));
  const allowMatching = builder.dataset.allowMatching === "true";
  const selected = { face_up: null, face_down: null };
  const otherSlot = { face_up: "face_down", face_down: "face_up" };

  function cardAt(index) {
    return index === null ? null : cards[index];
  }

  function updatePreview(slot) {
    const card = cardAt(selected[slot]);
    const image = builder.querySelector(`[data-preview-image="${slot}"]`);
    const placeholder = builder.querySelector(`[data-preview-placeholder="${slot}"]`);
    if (!(image instanceof HTMLImageElement) || !(placeholder instanceof HTMLElement)) return;

    image.hidden = card === null;
    placeholder.hidden = card !== null;
    if (card !== null && slot === "face_up") {
      image.src = card.dataset.cardImage || "";
      image.alt = card.dataset.cardAlt || "Face-up card";
    }
  }

  function update() {
    for (const [index, card] of cards.entries()) {
      const cardIndex = Number(card.dataset.cardIndex ?? index);
      card.classList.toggle(
        "is-selected",
        selected.face_up === cardIndex || selected.face_down === cardIndex,
      );

      for (const button of card.querySelectorAll("[data-select-slot]")) {
        if (!(button instanceof HTMLButtonElement)) continue;
        const slot = button.dataset.selectSlot;
        if (slot !== "face_up" && slot !== "face_down") continue;

        const isSelected = selected[slot] === cardIndex;
        const selectedForSlot = selected[slot];
        const oppositeCard = cardAt(selected[otherSlot[slot]]);
        const samePhysicalCard = selected[otherSlot[slot]] === cardIndex;
        const matchingName =
          oppositeCard !== null && oppositeCard.dataset.cardName === card.dataset.cardName;
        button.disabled =
          !isSelected &&
          (selectedForSlot !== null || samePhysicalCard || (!allowMatching && matchingName));
        button.classList.toggle("is-undo", isSelected);
        button.setAttribute("aria-pressed", String(isSelected));

        const label = button.querySelector("[data-selection-label]");
        const undo = button.querySelector("[data-undo-label]");
        if (label instanceof HTMLElement) label.hidden = isSelected;
        if (undo instanceof HTMLElement) undo.hidden = !isSelected;
      }
    }

    for (const slot of ["face_up", "face_down"]) {
      const input = builder.querySelector(`[data-offer-value="${slot}"]`);
      const card = cardAt(selected[slot]);
      if (input instanceof HTMLInputElement) input.value = card?.dataset.cardName || "";
      updatePreview(slot);
    }

    const complete = selected.face_up !== null && selected.face_down !== null;
    const confirm = builder.querySelector("[data-confirm-offer]");
    const status = builder.querySelector("[data-selection-status]");
    if (confirm instanceof HTMLButtonElement) confirm.disabled = !complete;
    if (status instanceof HTMLElement) {
      status.textContent = complete
        ? "Offer ready. Confirm to play these cards."
        : "Choose both cards to continue.";
    }
  }

  builder.addEventListener("click", (event) => {
    const target = event.target;
    if (!(target instanceof Element)) return;
    const button = target.closest("[data-select-slot]");
    if (!(button instanceof HTMLButtonElement) || button.disabled) return;
    const card = button.closest("[data-hand-card]");
    const slot = button.dataset.selectSlot;
    if (!(card instanceof HTMLElement) || (slot !== "face_up" && slot !== "face_down")) return;

    const index = Number(card.dataset.cardIndex);
    selected[slot] = selected[slot] === index ? null : index;
    update();
  });

  builder.addEventListener("submit", (event) => {
    if (selected.face_up === null || selected.face_down === null) event.preventDefault();
  });

  update();
})();
