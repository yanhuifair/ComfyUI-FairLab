import { api } from "/scripts/api.js";
import { app } from "/scripts/app.js";

const EXTENSION_NAME =
  "fairlab.web";
const RESTART_BUTTON_ID =
  "fairlab-restart-button";
let executedListenerRegistered = false;

function buildImageUrl(
  params,
  node,
) {
  const query =
    new URLSearchParams(
      params,
    ).toString();
  return api.apiURL(
    "/view?" +
      query +
      (node.animatedImages
        ? ""
        : app.getPreviewFormatParam()) +
      app.getRandParam(),
  );
}

function getNodeById(
  nodeId,
) {
  const numericNodeId =
    Number(
      nodeId,
    );

  if (
    Number.isFinite(
      numericNodeId,
    ) &&
    app
      .graph
      ?.getNodeById
  ) {
    return app.graph.getNodeById(
      numericNodeId,
    );
  }

  return (
    app
      .graph
      ?._nodes_by_id?.[
      nodeId
    ] ??
    null
  );
}

function downloadImages(
  images,
) {
  for (const image of images) {
    if (
      !image?.src
    ) {
      continue;
    }

    const anchor =
      document.createElement(
        "a",
      );
    const url =
      new URL(
        image.src,
      );

    url.searchParams.delete(
      "preview",
    );
    anchor.href =
      url;
    anchor.setAttribute(
      "download",
      new URLSearchParams(
        url.search,
      ).get(
        "filename",
      ) ??
        "image",
    );
    document.body.append(
      anchor,
    );
    anchor.click();
    requestAnimationFrame(
      () =>
        anchor.remove(),
    );
  }
}

async function handleExecuted(
  event,
) {
  const detail =
    event?.detail;
  const node =
    getNodeById(
      detail?.node,
    );

  if (
    node?.comfyClass !==
    "DownloadImageNode"
  ) {
    return;
  }

  const imageParams =
    detail
      ?.output
      ?.images;
  if (
    !Array.isArray(
      imageParams,
    ) ||
    imageParams.length ===
      0
  ) {
    return;
  }

  const loadedImages =
    await Promise.all(
      imageParams.map(
        async (
          params,
        ) => {
          const src =
            buildImageUrl(
              params,
              node,
            );

          return await new Promise(
            (
              resolve,
            ) => {
              const image =
                new Image();
              image.onload =
                () =>
                  resolve(
                    image,
                  );
              image.onerror =
                () =>
                  resolve(
                    null,
                  );
              image.src =
                src;
            },
          );
        },
      ),
    );

  downloadImages(
    loadedImages.filter(
      Boolean,
    ),
  );
}

const extension =
  {
    name: EXTENSION_NAME,

    async setup() {
      if (
        !executedListenerRegistered
      ) {
        api.addEventListener(
          "executed",
          handleExecuted,
        );
        executedListenerRegistered = true;
      }

      if (
        !app
          .menu
          ?.settingsGroup ||
        document.getElementById(
          RESTART_BUTTON_ID,
        )
      ) {
        return;
      }

      try {
        const {
          ComfyButton,
        } =
          await import("/scripts/ui/components/button.js");
        const restartButton =
          new ComfyButton(
            {
              icon: "restart",
              action:
                async () => {
                  await api.fetchApi(
                    "/manager/reboot",
                  );
                },
              tooltip:
                "Restart the server",
              content:
                "Restart",
            },
          );
        restartButton.enabled = true;
        restartButton.element.style.display =
          "";
        restartButton.element.id =
          RESTART_BUTTON_ID;
        app.menu.settingsGroup.append(
          restartButton,
        );
      } catch (error) {
        console.warn(
          "FairLab: failed to attach restart button",
          error,
        );
      }
    },
  };

app.registerExtension(
  extension,
);
