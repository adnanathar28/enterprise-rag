import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { UploadPanel } from "./UploadPanel";

const capabilities = {
  providers: [],
  allowed_document_extensions: [".pdf"],
  max_upload_size_bytes: 100,
};

test("selects and removes a PDF locally without implying it was uploaded", async () => {
  render(<UploadPanel capabilities={capabilities} />);
  const file = new File(["example"], "Supplier_Agreement.PDF", { type: "application/pdf" });
  fireEvent.change(screen.getByLabelText("PDF document"), { target: { files: [file] } });
  expect(screen.getByText(file.name)).toHaveAttribute("title", file.name);
  expect(screen.getByRole("status")).toHaveTextContent("Selected locally · Not uploaded");
  await userEvent.click(screen.getByRole("button", { name: "Remove" }));
  expect(screen.queryByRole("status")).not.toBeInTheDocument();
});

test.each([
  [[new File(["text"], "handbook.docx")], "Choose a PDF document."],
  [[new File([], "empty.pdf")], "This file is empty. Choose another PDF."],
  [[new File(["x".repeat(101)], "large.pdf")], "This file exceeds the upload size limit."],
  [[new File(["a"], "a.pdf"), new File(["b"], "b.pdf")], "Choose one PDF at a time."],
])("rejects invalid drops: %s", (files, message) => {
  render(<UploadPanel capabilities={capabilities} />);
  fireEvent.drop(screen.getByText("Drop a document here"), { dataTransfer: { files } });
  expect(screen.getByRole("alert")).toHaveTextContent(message);
  expect(screen.queryByRole("status")).not.toBeInTheDocument();
});

test("accepts a PDF drop and respects the server's PDF capability", () => {
  const { rerender } = render(<UploadPanel capabilities={capabilities} />);
  const file = new File(["pdf"], "research.pdf");
  fireEvent.drop(screen.getByText("Drop a document here"), { dataTransfer: { files: [file] } });
  expect(screen.getByRole("status")).toHaveTextContent(file.name);
  rerender(<UploadPanel capabilities={{ ...capabilities, allowed_document_extensions: [] }} />);
  expect(screen.getByRole("button", { name: "Choose a PDF" })).toBeDisabled();
  expect(screen.getByText("PDF uploads are unavailable on this server.")).toBeVisible();
});
