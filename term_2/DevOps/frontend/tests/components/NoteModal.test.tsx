// frontend/tests/components/NoteModal.test.tsx
import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import NoteModal from '../../components/NoteModal';
import { Note, Area } from '../../types';

const mockNote: Note = {
  id: '1',
  title: 'Test Note',
  content: 'Test content',
  area_id: '1',
  updated_at: new Date().toISOString(),
};

const mockAreas: Area[] = [
  { id: '1', name: 'Work', color: 'bg-red-500' },
  { id: '2', name: 'Personal', color: 'bg-blue-500' },
];

describe('NoteModal Component', () => {
  const onClose = vi.fn();
  const onSave = vi.fn();

  beforeEach(() => {
    onClose.mockClear();
    onSave.mockClear();
  });

  it('renders correctly when open with a note', () => {
    render(
      <NoteModal
        isOpen={true}
        note={mockNote}
        onClose={onClose}
        onSave={onSave}
        areas={mockAreas}
      />
    );

    expect(screen.getByPlaceholderText('Untitled Note')).toHaveValue(mockNote.title);
    expect(screen.getByPlaceholderText('Start writing...')).toHaveValue(mockNote.content);
    expect(screen.getByDisplayValue('Work')).toBeInTheDocument();
  });

  it('does not render when closed', () => {
    const { container } = render(
      <NoteModal
        isOpen={false}
        note={mockNote}
        onClose={onClose}
        onSave={onSave}
        areas={mockAreas}
      />
    );
    expect(container.firstChild).toBeNull();
  });

  it('calls onClose when the close button is clicked', async () => {
    render(
        <NoteModal
          isOpen={true}
          note={mockNote}
          onClose={onClose}
          onSave={onSave}
          areas={mockAreas}
        />
      );

    await userEvent.click(screen.getByText('close'));
    expect(onClose).toHaveBeenCalled();
  });

  it('updates form data on input change', async () => {
    render(
        <NoteModal
          isOpen={true}
          note={mockNote}
          onClose={onClose}
          onSave={onSave}
          areas={mockAreas}
        />
      );

    const titleInput = screen.getByPlaceholderText('Untitled Note');
    await userEvent.clear(titleInput);
    await userEvent.type(titleInput, 'New Title');
    expect(titleInput).toHaveValue('New Title');

    const contentTextarea = screen.getByPlaceholderText('Start writing...');
    await userEvent.clear(contentTextarea);
    await userEvent.type(contentTextarea, 'New Content');
    expect(contentTextarea).toHaveValue('New Content');
  });

  it('calls onSave with updated data when Done is clicked', async () => {
    render(
        <NoteModal
          isOpen={true}
          note={mockNote}
          onClose={onClose}
          onSave={onSave}
          areas={mockAreas}
        />
      );

    const titleInput = screen.getByPlaceholderText('Untitled Note');
    await userEvent.clear(titleInput);
    await userEvent.type(titleInput, 'Updated Title');
    await userEvent.click(screen.getByText('Done'));

    expect(onSave).toHaveBeenCalledWith({
      id: mockNote.id,
      title: 'Updated Title',
      content: mockNote.content,
      area_id: mockNote.area_id,
    });
    expect(onClose).toHaveBeenCalled();
  });

  it('does not call onSave if title is empty', async () => {
    render(
        <NoteModal
          isOpen={true}
          note={mockNote}
          onClose={onClose}
          onSave={onSave}
          areas={mockAreas}
        />
      );

    const titleInput = screen.getByPlaceholderText('Untitled Note');
    await userEvent.clear(titleInput);
    await userEvent.click(screen.getByText('Done'));

    expect(onSave).not.toHaveBeenCalled();
    expect(onClose).not.toHaveBeenCalled();
  });
});
