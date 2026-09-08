import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { vi } from "vitest";

import { UploadPanel } from "./UploadPanel";

const capabilities = {
  providers: [],
  allowed_document_extensions: [".pdf"],
  max_upload_size_bytes: 100,
};

const baseProps = {
  capabilities,
  phase: null,
  processingFilename: null,
  uploadError: null,
  onUpload: vi.fn(),
} as const;

test("selects and removes a PDF locally without implying it was uploaded", async () => {
  const onUpload = vi.fn();
  render(<UploadPanel {...baseProps} onUpload={onUpload} />);
  const file = new File(["example"], "Supplier_Agreement.PDF", { type: "application/pdf" });
  fireEvent.change(screen.getByLabelText("PDF document"), { target: { files: [file] } });
  expect(onUpload).toHaveBeenCalledWith(file);
  expect(screen.getByText(file.name)).toHaveAttribute("title", file.name);
  expect(screen.getByRole("status")).toHaveTextContent("Selected");
  await userEvent.click(screen.getByRole("button", { name: "Remove" }));
  expect(screen.queryByRole("status")).not.toBeInTheDocument();
});

test.each([
  [[new File(["text"], "handbook.docx")], "Choose a PDF document."],
  [[new File([], "empty.pdf")], "This file is empty. Choose another PDF."],
  [[new File(["x".repeat(101)], "large.pdf")], "This file exceeds the upload size limit."],
  [[new File(["a"], "a.pdf"), new File(["b"], "b.pdf")], "Choose one PDF at a time."],
])("rejects invalid drops: %s", (files, message) => {
  render(<UploadPanel {...baseProps} />);
  fireEvent.drop(screen.getByText("Drop a document here"), { dataTransfer: { files } });
  expect(screen.getByRole("alert")).toHaveTextContent(message);
  expect(screen.queryByRole("status")).not.toBeInTheDocument();
});

test("accepts a PDF drop and respects the server's PDF capability", () => {
  const onUpload = vi.fn();
  const { rerender } = render(<UploadPanel {...baseProps} onUpload={onUpload} />);
  const file = new File(["pdf"], "research.pdf");
  fireEvent.drop(screen.getByText("Drop a document here"), { dataTransfer: { files: [file] } });
  expect(screen.getByRole("status")).toHaveTextContent(file.name);
  expect(onUpload).toHaveBeenCalledWith(file);
  rerender(<UploadPanel {...baseProps} capabilities={{ ...capabilities, allowed_document_extensions: [] }} />);
  expect(screen.getByRole("button", { name: "Choose a PDF" })).toBeDisabled();
  expect(screen.getByText("PDF uploads are unavailable on this server.")).toBeVisible();
});

test("shows honest processing stages", () => {
  const { rerender } = render(
    <UploadPanel {...baseProps} phase="uploading" processingFilename="report.pdf" />,
  );
  expect(screen.getByRole("heading", { name: "Uploading and parsing document" })).toBeVisible();
  expect(screen.getByText("Upload and parse")).toHaveAttribute("data-state", "active");
  rerender(<UploadPanel {...baseProps} phase="indexing" processingFilename="report.pdf" />);
  expect(screen.getByRole("heading", { name: "Preparing for search" })).toBeVisible();
  expect(screen.getByText("Upload and parse")).toHaveAttribute("data-state", "complete");
  expect(screen.getByText("Prepare for search")).toHaveAttribute("data-state", "active");
});

test("allows retrying a failed upload with the selected file", async () => {
  const onUpload = vi.fn();
  const { rerender } = render(<UploadPanel {...baseProps} onUpload={onUpload} />);
  const file = new File(["pdf"], "report.pdf");
  fireEvent.change(screen.getByLabelText("PDF document"), { target: { files: [file] } });
  rerender(
    <UploadPanel {...baseProps} uploadError="The document could not be parsed." onUpload={onUpload} />,
  );
  expect(screen.getByRole("alert")).toHaveTextContent("The document could not be parsed.");
  await userEvent.click(screen.getByRole("button", { name: "Try again" }));
  expect(onUpload).toHaveBeenCalledTimes(2);
});
